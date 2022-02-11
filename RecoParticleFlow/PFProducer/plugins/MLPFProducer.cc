#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"

using namespace cms::Ort;

//use this to switch on detailed print statements in MLPF
//#define MLPF_DEBUG

class MLPFProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit MLPFProducer(const edm::ParameterSet&, const ONNXRuntime*);

  void produce(edm::Event& event, const edm::EventSetup& setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // static methods for handling the global cache
  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  const edm::EDPutTokenT<reco::PFCandidateCollection> pfCandidatesPutToken_;
  const edm::EDGetTokenT<reco::PFBlockCollection> inputTagBlocks_;
};

MLPFProducer::MLPFProducer(const edm::ParameterSet& cfg, const ONNXRuntime* cache)
    : pfCandidatesPutToken_{produces<reco::PFCandidateCollection>()},
      inputTagBlocks_(consumes<reco::PFBlockCollection>(cfg.getParameter<edm::InputTag>("src"))) {}

void MLPFProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  using namespace reco::mlpf;

  const auto& blocks = event.get(inputTagBlocks_);
  const auto& all_elements = getPFElements(blocks);

  std::vector<const reco::PFBlockElement*> selected_elements;
  unsigned int num_elements_total = 0;
  for (const auto* pelem : all_elements) {
    if (pelem->type() == reco::PFBlockElement::PS1 || pelem->type() == reco::PFBlockElement::PS2) {
      continue;
    }
    num_elements_total += 1;
    selected_elements.push_back(pelem);
  }
  const auto tensor_size = LSH_BIN_SIZE * std::max(2u, (num_elements_total / LSH_BIN_SIZE + 1));

#ifdef MLPF_DEBUG
  assert(num_elements_total < NUM_MAX_ELEMENTS_BATCH);
  //tensor size must be a multiple of the bin size and larger than the number of elements
  assert(tensor_size <= NUM_MAX_ELEMENTS_BATCH);
  assert(tensor_size % LSH_BIN_SIZE == 0);
#endif

#ifdef MLPF_DEBUG
  std::cout << "tensor_size=" << tensor_size << std::endl;
#endif

  //Fill the input tensor (batch, elems, features) = (1, tensor_size, NUM_ELEMENT_FEATURES)
  std::vector<std::vector<float>> inputs(1, std::vector<float>(NUM_ELEMENT_FEATURES * tensor_size, 0.0));
  unsigned int ielem = 0;
  for (const auto* pelem : selected_elements) {
    if (ielem > tensor_size) {
      continue;
    }

    const auto& elem = *pelem;

    //prepare the input array from the PFElement
    const auto& props = getElementProperties(elem);

    //copy features to the input array
    for (unsigned int iprop = 0; iprop < NUM_ELEMENT_FEATURES; iprop++) {
      inputs[0][ielem * NUM_ELEMENT_FEATURES + iprop] = normalize(props[iprop]);
    }
    ielem += 1;
  }

  //run the GNN inference, given the inputs and the output.
  const auto& outputs = globalCache()->run({"x:0"}, inputs, {{1, tensor_size, NUM_ELEMENT_FEATURES}});
  const auto& output = outputs[0];
#ifdef MLPF_DEBUG
  assert(output.size() == tensor_size * NUM_OUTPUT_FEATURES);
#endif

  std::vector<reco::PFCandidate> pOutputCandidateCollection;
  for (size_t ielem = 0; ielem < num_elements_total; ielem++) {
    std::vector<float> pred_id_probas(IDX_CLASS + 1, 0.0);
    const reco::PFBlockElement* elem = selected_elements[ielem];

    for (unsigned int idx_id = 0; idx_id <= IDX_CLASS; idx_id++) {
      auto pred_proba = output[ielem * NUM_OUTPUT_FEATURES + idx_id];
#ifdef MLPF_DEBUG
      assert(!std::isnan(pred_proba));
#endif
      pred_id_probas[idx_id] = pred_proba;
    }

    auto imax = argMax(pred_id_probas);

    //get the most probable class PDGID
    int pred_pid = pdgid_encoding[imax];

#ifdef MLPF_DEBUG
    std::cout << "ielem=" << ielem << " inputs:";
    for (unsigned int iprop = 0; iprop < NUM_ELEMENT_FEATURES; iprop++) {
      std::cout << iprop << "=" << inputs[0][ielem * NUM_ELEMENT_FEATURES + iprop] << " ";
    }
    std::cout << std::endl;
    std::cout << "ielem=" << ielem << " pred: pid=" << pred_pid << std::endl;
#endif

    //a particle was predicted for this PFElement, otherwise it was a spectator
    if (pred_pid != 0) {
      //muons and charged hadrons should only come from tracks, otherwise we won't have track references to pass downstream
      if (((pred_pid == 13) || (pred_pid == 211)) && elem->type() != reco::PFBlockElement::TRACK) {
        pred_pid = 130;
      }

      if (elem->type() == reco::PFBlockElement::TRACK) {
        const auto* eltTrack = dynamic_cast<const reco::PFBlockElementTrack*>(elem);

        //a track with no muon ref should not produce a muon candidate, instead we interpret it as a charged hadron
        if (pred_pid == 13 && eltTrack->muonRef().isNull()) {
          pred_pid = 211;
        }

        //tracks from displaced vertices need reference debugging downstream as well, so we just treat them as neutrals for the moment
        if ((pred_pid == 211) && (eltTrack->isLinkedToDisplacedVertex())) {
          pred_pid = 130;
        }
      }

      //get the predicted momentum components
      float pred_pt = output[ielem * NUM_OUTPUT_FEATURES + IDX_PT];
      float pred_eta = output[ielem * NUM_OUTPUT_FEATURES + IDX_ETA];
      float pred_sin_phi = output[ielem * NUM_OUTPUT_FEATURES + IDX_SIN_PHI];
      float pred_cos_phi = output[ielem * NUM_OUTPUT_FEATURES + IDX_COS_PHI];
      float pred_e = output[ielem * NUM_OUTPUT_FEATURES + IDX_ENERGY];
      float pred_charge = output[ielem * NUM_OUTPUT_FEATURES + IDX_CHARGE];

      auto cand = makeCandidate(pred_pid, pred_charge, pred_pt, pred_eta, pred_sin_phi, pred_cos_phi, pred_e);
      setCandidateRefs(cand, selected_elements, ielem);
      pOutputCandidateCollection.push_back(cand);

#ifdef MLPF_DEBUG
      std::cout << "ielem=" << ielem << " cand: pid=" << cand.pdgId() << " E=" << cand.energy() << " pt=" << cand.pt()
                << " eta=" << cand.eta() << " phi=" << cand.phi() << " charge=" << cand.charge() << std::endl;
#endif
    }
  }  //loop over PFElements

  event.emplace(pfCandidatesPutToken_, pOutputCandidateCollection);
}

std::unique_ptr<ONNXRuntime> MLPFProducer::initializeGlobalCache(const edm::ParameterSet& params) {
  return std::make_unique<ONNXRuntime>(params.getParameter<edm::FileInPath>("model_path").fullPath());
}

void MLPFProducer::globalEndJob(const ONNXRuntime* cache) {}

void MLPFProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("particleFlowBlock"));
  desc.add<edm::FileInPath>(
      "model_path",
      edm::FileInPath(
          "RecoParticleFlow/PFProducer/data/mlpf/"
          "mlpf_2021_11_16__no_einsum__all_data_cms-best-of-asha-scikit_20211026_042043_178263.workergpu010.onnx"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MLPFProducer);
