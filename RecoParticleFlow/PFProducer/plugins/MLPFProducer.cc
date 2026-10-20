#include <cmath>
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "FWCore/AbstractServices/interface/ResourceInformation.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"

using namespace cms::Ort;
using namespace reco::mlpf;

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
  const edm::EDGetTokenT<edm::View<reco::GsfElectron>> gsfElectrons_;
  const edm::EDGetTokenT<reco::PFBlockCollection> inputTagBlocks_;

  const bool noRegressionEndcap_;
  const StringCutObjectSelector<reco::Track> endcapTk_;

  const bool additionalTrackFilterNoHCAL_;
  const AdditionalTrackFilterParams additionalTrackFilterParams_;

  using ElementMap = std::multimap<double, unsigned>;
};

MLPFProducer::MLPFProducer(const edm::ParameterSet& cfg, const ONNXRuntime* cache)
    : pfCandidatesPutToken_{produces<reco::PFCandidateCollection>()},
      gsfElectrons_{consumes<edm::View<reco::GsfElectron>>(edm::InputTag("gedGsfElectronsTmp"))},
      inputTagBlocks_{consumes<reco::PFBlockCollection>(cfg.getParameter<edm::InputTag>("src"))},
      noRegressionEndcap_(cfg.getParameter<bool>("noRegressionEndcap")),
      endcapTk_(cfg.getParameter<std::string>("endcapTk")),
      additionalTrackFilterNoHCAL_(cfg.getParameter<bool>("additionalTrackFilterNoHCAL")),
      additionalTrackFilterParams_{
          std::make_unique<StringCutObjectSelector<reco::Track>>(
              cfg.getParameter<edm::ParameterSet>("AdditionalTrackFilterNoHCALParams")
                  .getParameter<std::string>("additionalFilterTk")),
          cfg.getParameter<edm::ParameterSet>("AdditionalTrackFilterNoHCALParams").getParameter<double>("pt_Error")} {}

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

  //Fill the input tensor (batch, elems, features) = (1, tensor_size, NUM_ELEMENT_FEATURES)
  std::vector<std::vector<float>> inputs;
  inputs.push_back(std::vector<float>(NUM_ELEMENT_FEATURES * tensor_size, 0.0));
  inputs.push_back(std::vector<float>(tensor_size, 0.0));
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

  //run the GNN inference, given the inputs and the output.
  const auto& outputs = globalCache()->run(
      {"Xfeat_normed", "mask"},
      inputs,
      {{1, static_cast<long int>(tensor_size), NUM_ELEMENT_FEATURES}, {1, static_cast<long int>(tensor_size)}});
  const auto& output_binary = outputs[0];
  const auto& output_pid = outputs[1];
  const auto& output_p4 = outputs[2];

#ifdef MLPF_DEBUG
  std::cout << "output_binary=" << output_binary.size() << std::endl;
  assert(output_binary.size() == tensor_size * 2);

  std::cout << "output_pid=" << output_pid.size() << std::endl;
  assert(output_pid.size() == tensor_size * NUM_OUTPUT_FEATURES_CLS);

  std::cout << "output_p4=" << output_p4.size() << std::endl;
  assert(output_p4.size() == tensor_size * NUM_OUTPUT_FEATURES_P4);
#endif

  auto candidates = makeCandidates(inputs,
                                   output_binary,
                                   output_pid,
                                   output_p4,
                                   blocks,
                                   additionalTrackFilterNoHCAL_,
                                   additionalTrackFilterParams_,
                                   noRegressionEndcap_,
                                   endcapTk_,
                                   selected_elements);
  event.emplace(pfCandidatesPutToken_, candidates);
}

std::unique_ptr<ONNXRuntime> MLPFProducer::initializeGlobalCache(const edm::ParameterSet& params) {
  edm::Service<edm::ResourceInformation> ri;

  Backend backend = Backend::cpu;

  if (ri.isAvailable() && ri->hasGpuNvidia()) {
    backend = Backend::cuda;
    edm::LogInfo("MLPFProducer") << "NVIDIA GPU detected. Running ONNX model on CUDA.";
  } else {
    edm::LogInfo("MLPFProducer") << "No NVIDIA GPU detected. Running ONNX model on CPU.";
  }

  auto session_options = ONNXRuntime::defaultSessionOptions(backend);
  return std::make_unique<ONNXRuntime>(params.getParameter<edm::FileInPath>("model_path").fullPath(), &session_options);
}

void MLPFProducer::globalEndJob(const ONNXRuntime* cache) {}

void MLPFProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("particleFlowBlock"));
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("RecoParticleFlow/PFProducer/data/mlpf/model.onnx"));
  //
  desc.add<bool>("additionalTrackFilterNoHCAL", false)
      ->setComment("Apply tight requirments for tracks not linked to HCAL clusters.");
  {
    edm::ParameterSetDescription psd;
    psd.add<std::string>("additionalFilterTk", "1.48 < abs(eta) < 3.0");
    psd.add<double>("pt_Error", 1.0)->setComment("Absolute pt error to detect fake tracks.");
    desc.add<edm::ParameterSetDescription>("AdditionalTrackFilterNoHCALParams", psd);
  }
  //
  desc.add<bool>("noRegressionEndcap", false)->setComment("Turn on/off regression for tracks in the endcap.");
  desc.add<std::string>("endcapTk", "1.48 < abs(eta) < 3.0");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MLPFProducer);
