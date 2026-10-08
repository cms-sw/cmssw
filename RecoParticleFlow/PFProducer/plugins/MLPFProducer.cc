#include <atomic>
#include <cmath>
#include <memory>
#include <optional>

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXInterface.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "PhysicsTools/ONNXRuntime/interface/SessionCache.h"
#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"

using namespace cms::Ort;

//use this to switch on detailed print statements in MLPF
//#define MLPF_DEBUG

// The ONNX Runtime sessions used by MLPFProducer.
//
// MIGraphX compiles the model for the shape of its inputs, and recompiles it whenever the shape changes, which happens
// at almost every event. So on ROCm the inputs are padded to a fixed number of elements, rocmMaxElements, and the
// events with more elements run instead on a CPU session, created only if one such event is found.
struct MLPFSessions {
  MLPFSessions(std::string const& model_path, Backend backend, unsigned int maxElements)
      : sessions(model_path, backend), maxElements(maxElements) {
    if (backend == Backend::rocm) {
      fallback.emplace(model_path, Backend::cpu);
    }
  }

  SessionCache sessions;
  std::optional<SessionCache> fallback;  // only on ROCm; its session is created on first use
  const unsigned int maxElements;        // only on ROCm: the inputs are padded to this size
  mutable std::atomic<unsigned int> events = 0;
  mutable std::atomic<unsigned int> fallbackEvents = 0;
};

class MLPFProducer : public edm::stream::EDProducer<edm::GlobalCache<MLPFSessions>> {
public:
  explicit MLPFProducer(const edm::ParameterSet&, const MLPFSessions*);

  // Create the ONNX Runtime session used by this framework stream before the first event: on the CPU all the streams
  // share the same session, on a GPU each stream has its own, running in its own compute stream.
  void beginStream(edm::StreamID id) override;

  void produce(edm::Event& event, const edm::EventSetup& setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // static methods for handling the global cache
  static std::unique_ptr<MLPFSessions> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const MLPFSessions*);

private:
  const edm::EDPutTokenT<reco::PFCandidateCollection> pfCandidatesPutToken_;
  const edm::EDGetTokenT<edm::View<reco::GsfElectron>> gsfElectrons_;
  const edm::EDGetTokenT<reco::PFBlockCollection> inputTagBlocks_;
};

MLPFProducer::MLPFProducer(const edm::ParameterSet& cfg, const MLPFSessions* cache)
    : pfCandidatesPutToken_{produces<reco::PFCandidateCollection>()},
      gsfElectrons_{consumes<edm::View<reco::GsfElectron>>(edm::InputTag("gedGsfElectronsTmp"))},
      inputTagBlocks_{consumes<reco::PFBlockCollection>(cfg.getParameter<edm::InputTag>("src"))} {}

void MLPFProducer::beginStream(edm::StreamID id) { globalCache()->sessions.get(id); }

void MLPFProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  using namespace reco::mlpf;

  const auto& blocks = event.get(inputTagBlocks_);
  const auto& all_elements = getPFElements(blocks);

  const auto& gsfElectrons = event.get(gsfElectrons_);

  std::vector<const reco::PFBlockElement*> selected_elements;
  for (const auto* pelem : all_elements) {
    if (pelem->type() == reco::PFBlockElement::PS1 || pelem->type() == reco::PFBlockElement::PS2 ||
        pelem->type() == reco::PFBlockElement::BREM) {
      continue;
    }
    selected_elements.push_back(pelem);
  }

  const auto tensor_size = selected_elements.size();

  // On ROCm, pad the inputs to the fixed size, or run on the CPU if the event has more elements
  const MLPFSessions& cache = *globalCache();
  const bool fallback = cache.fallback and tensor_size > cache.maxElements;
  const std::size_t input_size = (cache.fallback and not fallback) ? cache.maxElements : tensor_size;
  ++cache.events;
  if (fallback) {
    ++cache.fallbackEvents;
  }

  //Fill the input tensor (batch, elems, features) = (1, input_size, NUM_ELEMENT_FEATURES)
  //the padding elements beyond tensor_size are zero, and masked
  std::vector<std::vector<float>> inputs;
  inputs.push_back(std::vector<float>(NUM_ELEMENT_FEATURES * input_size, 0.0));
  inputs.push_back(std::vector<float>(input_size, 0.0));
  unsigned int ielem = 0;
  for (const auto* pelem : selected_elements) {
    if (ielem > tensor_size) {
      continue;
    }
#ifdef MLPF_DEBUG
    std::cout << "ielem=" << ielem << std::endl;
#endif

    const auto& elem = *pelem;

    //prepare the input array from the PFElement
    const auto& props = getElementProperties(elem, gsfElectrons).as_array();

    //copy features to the input array
    for (unsigned int iprop = 0; iprop < NUM_ELEMENT_FEATURES; iprop++) {
      const auto vec_elem = ielem * NUM_ELEMENT_FEATURES + iprop;
      assert(vec_elem < inputs[0].size());
      inputs[0][vec_elem] = normalize(props[iprop]);
    }
    //mask
    inputs[1][ielem] = 1.0;
    ielem += 1;
  }

#ifdef MLPF_DEBUG
  for (unsigned int _idx = 0; _idx < inputs[0].size(); _idx++) {
    std::cout << inputs[0][_idx] << " ";
  }
  std::cout << std::endl;
#endif

  //run the GNN inference, given the inputs and the output; the outputs of the padding elements are ignored
  const SessionCache& sessions = fallback ? *cache.fallback : cache.sessions;
  const auto& outputs =
      sessions.get(event.streamID())
          .run({"Xfeat_normed", "mask"},
               inputs,
               {{1, static_cast<long int>(input_size), NUM_ELEMENT_FEATURES}, {1, static_cast<long int>(input_size)}});
  const auto& output_binary = outputs[0];
  const auto& output_pid = outputs[1];
  const auto& output_p4 = outputs[2];

#ifdef MLPF_DEBUG
  std::cout << "output_binary=" << output_binary.size() << std::endl;
  assert(output_binary.size() == input_size * 2);

  std::cout << "output_pid=" << output_pid.size() << std::endl;
  assert(output_pid.size() == input_size * NUM_OUTPUT_FEATURES_CLS);

  std::cout << "output_p4=" << output_p4.size() << std::endl;
  assert(output_p4.size() == input_size * NUM_OUTPUT_FEATURES_P4);
#endif

  std::vector<reco::PFCandidate> pOutputCandidateCollection;
  for (size_t ielem = 0; ielem < selected_elements.size(); ielem++) {
    std::vector<float> pred_id_probas(pdgid_encoding.size(), 0.0);
    const reco::PFBlockElement* elem = selected_elements[ielem];

#ifdef MLPF_DEBUG
    std::cout << "ielem=" << ielem << " inputs:";
    for (unsigned int iprop = 0; iprop < NUM_ELEMENT_FEATURES; iprop++) {
      std::cout << iprop << "=" << inputs[0][ielem * NUM_ELEMENT_FEATURES + iprop] << " ";
    }
    std::cout << std::endl;
#endif

    const auto logit_no_ptcl = output_binary[ielem * 2 + 0];
    const auto logit_ptcl = output_binary[ielem * 2 + 1];
#ifdef MLPF_DEBUG
    std::cout << "binary: " << logit_no_ptcl << " " << logit_ptcl << std::endl;
#endif

    // Check if the binary classifier of the model predicted a particle
    int pred_pid = 0;
    if (logit_ptcl > logit_no_ptcl) {
      for (unsigned int idx_id = 0; idx_id < pred_id_probas.size(); idx_id++) {
        auto pred_proba = output_pid[ielem * NUM_OUTPUT_FEATURES_CLS + idx_id];
#ifdef MLPF_DEBUG
        std::cout << "pid proba: " << pred_proba << std::endl;
        assert(!std::isnan(pred_proba));
#endif
        pred_id_probas[idx_id] = pred_proba;
      }

      auto imax = argMax(pred_id_probas);

      //get the most probable class PDGID
      pred_pid = pdgid_encoding.at(imax);
#ifdef MLPF_DEBUG
      std::cout << "pid: " << pred_pid << std::endl;
#endif
    }

#ifdef MLPF_DEBUG
    std::cout << "p4: " << output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + 0] << " "
              << output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + 1] << " " << output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + 2]
              << " " << output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + 3] << " "
              << output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + 4] << std::endl;
#endif

    //a particle was predicted for this PFElement, otherwise it was a spectator
    if (pred_pid != 0) {
      //muons and charged hadrons should only come from tracks, otherwise we won't have track references to pass downstream
      if (((pred_pid == 13) || (pred_pid == 211)) && elem->type() != reco::PFBlockElement::TRACK) {
        pred_pid = 130;
      }

      float pred_charge = 0.0;
      if (elem->type() == reco::PFBlockElement::TRACK) {
        const auto* eltTrack = dynamic_cast<const reco::PFBlockElementTrack*>(elem);
        //for now, just take the charge from the track
        if (eltTrack->trackRef().isNonnull()) {
          pred_charge = eltTrack->trackRef()->charge();
        }

        //a track with no muon ref should not produce a muon candidate, instead we interpret it as a charged hadron here
        if (pred_pid == 13 && eltTrack->muonRef().isNull()) {
          pred_pid = 211;
        }

        //taus are reconstructed downstream based on other criteria, instead we interpret it as a charged hadron here
        if (pred_pid == 15) {
          pred_pid = 211;
        }

        //tracks from displaced vertices need reference debugging downstream as well, so we just treat them as neutrals for the moment
        if ((pred_pid == 211) && (eltTrack->isLinkedToDisplacedVertex())) {
          pred_pid = 130;
        }
      }

      //do not attempt to do PID in the HF
      if (elem->type() == reco::PFBlockElement::HFEM) {
        pred_pid = 2;
      } else if (elem->type() == reco::PFBlockElement::HFHAD) {
        pred_pid = 1;
      }

      //get the predicted momentum components from the model
      float pred_pt = output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + IDX_PT];
      pred_pt = exp(pred_pt) * inputs[0][ielem * NUM_ELEMENT_FEATURES + 1];
      float pred_eta = output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + IDX_ETA];
      float pred_sin_phi = output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + IDX_SIN_PHI];
      float pred_cos_phi = output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + IDX_COS_PHI];
      float pred_e = output_p4[ielem * NUM_OUTPUT_FEATURES_P4 + IDX_ENERGY];
      pred_e = exp(pred_e) * inputs[0][ielem * NUM_ELEMENT_FEATURES + 5];

      if (elem->type() == reco::PFBlockElement::TRACK) {
        const auto* eltTrack = dynamic_cast<const reco::PFBlockElementTrack*>(elem);
        if (eltTrack->trackRef().isNonnull()) {
          pred_eta = eltTrack->trackRef()->eta();
          pred_sin_phi = sin(eltTrack->trackRef()->phi());
          pred_cos_phi = cos(eltTrack->trackRef()->phi());
        }
      }

      auto cand = makeCandidate(pred_pid, pred_charge, pred_pt, pred_eta, pred_sin_phi, pred_cos_phi, pred_e);
      setCandidateRefs(cand, selected_elements, ielem);
      pOutputCandidateCollection.push_back(cand);

#ifdef MLPF_DEBUG
      std::cout << "ielem=" << ielem << " pred: pid=" << cand.pdgId() << " E=" << cand.energy() << " pt=" << cand.pt()
                << " eta=" << cand.eta() << " phi=" << cand.phi() << " charge=" << cand.charge() << std::endl;
#endif
    }
  }  //loop over PFElements

  event.emplace(pfCandidatesPutToken_, pOutputCandidateCollection);
}

std::unique_ptr<MLPFSessions> MLPFProducer::initializeGlobalCache(const edm::ParameterSet& params) {
  edm::Service<ONNXInterface> onnx;
  Backend backend = onnx->chooseBackend();
  auto cache = std::make_unique<MLPFSessions>(params.getParameter<edm::FileInPath>("model_path").fullPath(),
                                              backend,
                                              params.getParameter<unsigned int>("rocmMaxElements"));
  edm::LogInfo log("MLPFProducer");
  log << "Running the MLPF model on the " << backendName(backend) << " backend";
  if (cache->fallback) {
    log << ", up to " << cache->maxElements << " elements (padded), and on the CPU for larger events";
  }
  return cache;
}

void MLPFProducer::globalEndJob(const MLPFSessions* cache) {
  if (cache->fallback) {
    edm::LogInfo("MLPFProducer") << "MLPF ran " << cache->events - cache->fallbackEvents << " events on ROCm and "
                                 << cache->fallbackEvents << " events on the CPU fallback";
  }
}

void MLPFProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("particleFlowBlock"));
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("RecoParticleFlow/PFProducer/data/mlpf/mlpf_padded.onnx"))
      ->setComment("the model must support padded inputs (masked elements) to run on ROCm");
  desc.add<unsigned int>("rocmMaxElements", 4096)
      ->setComment(
          "ROCm only: maximum number of elements run on the GPU, where the inputs are padded to this size; larger "
          "events run on the CPU");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MLPFProducer);
