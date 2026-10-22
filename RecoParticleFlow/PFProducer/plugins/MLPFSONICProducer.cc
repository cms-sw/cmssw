#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"

#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"

class MLPFSONICProducer : public TritonEDProducer<> {
public:
  explicit MLPFSONICProducer(const edm::ParameterSet &);
  ~MLPFSONICProducer() override;

  void acquire(edm::Event const &iEvent, edm::EventSetup const &iSetup, Input &iInput) override;

  void produce(edm::Event &iEvent, edm::EventSetup const &iSetup, Output const &iOutput) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  const edm::EDPutTokenT<reco::PFCandidateCollection> pfCandidatesPutToken_;
  const edm::EDGetTokenT<edm::View<reco::GsfElectron>> gsfElectrons_;
  const edm::EDGetTokenT<reco::PFBlockCollection> inputTagBlocks_;

  const bool noRegressionEndcap_;
  const StringCutObjectSelector<reco::Track> endcapTk_;

  const bool additionalTrackFilterNoHCAL_;
  const reco::mlpf::AdditionalTrackFilterParams additionalTrackFilterParams_;

  std::vector<const reco::PFBlockElement *> selected_elements_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<float>> inputs;
};

MLPFSONICProducer::MLPFSONICProducer(const edm::ParameterSet &iConfig)
    : TritonEDProducer<>(iConfig),
      pfCandidatesPutToken_{produces<reco::PFCandidateCollection>()},
      gsfElectrons_{consumes<edm::View<reco::GsfElectron>>(edm::InputTag("gedGsfElectronsTmp"))},
      inputTagBlocks_{consumes<reco::PFBlockCollection>(iConfig.getParameter<edm::InputTag>("src"))},
      noRegressionEndcap_(iConfig.getParameter<bool>("noRegressionEndcap")),
      endcapTk_(iConfig.getParameter<std::string>("endcapTk")),
      additionalTrackFilterNoHCAL_(iConfig.getParameter<bool>("additionalTrackFilterNoHCAL")),
      additionalTrackFilterParams_{
          std::make_unique<StringCutObjectSelector<reco::Track>>(
              iConfig.getParameter<edm::ParameterSet>("AdditionalTrackFilterNoHCALParams")
                  .getParameter<std::string>("additionalFilterTk")),
          iConfig.getParameter<edm::ParameterSet>("AdditionalTrackFilterNoHCALParams").getParameter<double>("pt_Error")},
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {}

MLPFSONICProducer::~MLPFSONICProducer() {}
void MLPFSONICProducer::acquire(edm::Event const &iEvent, edm::EventSetup const &iSetup, Input &iInput) {
  using namespace reco::mlpf;
  const auto &blocks = iEvent.get(inputTagBlocks_);
  const auto &all_elements = getPFElements(blocks);

  const auto &gsfElectrons = iEvent.get(gsfElectrons_);

  selected_elements_.clear();
  for (const auto *pelem : all_elements) {
    if (pelem->type() == reco::PFBlockElement::PS1 || pelem->type() == reco::PFBlockElement::PS2 ||
        pelem->type() == reco::PFBlockElement::BREM) {
      continue;
    }
    selected_elements_.push_back(pelem);
  }

  const auto tensor_size = selected_elements_.size();

  //Fill the input tensor (batch, elems, features) = (1, tensor_size, NUM_ELEMENT_FEATURES)
  inputs.resize(2);
  inputs[0].assign(tensor_size * NUM_ELEMENT_FEATURES, 0.0);
  inputs[1].assign(tensor_size, 0.0);

  unsigned int ielem = 0;
  const auto &mask_name = input_names_[0];
  const auto &X_name = input_names_[1];

  auto &data1 = iInput.at(mask_name);
  auto &data2 = iInput.at(X_name);
  data1.setShape(0, tensor_size);
  data2.setShape(0, tensor_size);
  auto tdata1 = data1.allocate<float>(true);
  auto tdata2 = data2.allocate<float>(true);
  for (const auto *pelem : selected_elements_) {
    if (ielem > tensor_size) {
      continue;
    }

    const auto &elem = *pelem;

    //prepare the input array from the PFElement
    const auto &props = getElementProperties(elem, gsfElectrons).as_array();

    //copy features to the input array
    for (unsigned int iprop = 0; iprop < NUM_ELEMENT_FEATURES; iprop++) {
      inputs[0][ielem * NUM_ELEMENT_FEATURES + iprop] = normalize(props[iprop]);
    }
    //mask
    inputs[1][ielem] = 1.0;
    ielem += 1;
  }
  auto &vdata1 = (*tdata1)[0];
  auto &vdata2 = (*tdata2)[0];
  vdata1 = inputs[1];
  vdata2 = inputs[0];
  data1.toServer(tdata1);
  data2.toServer(tdata2);
}
void MLPFSONICProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup, Output const &iOutput) {
  using namespace reco::mlpf;
  const auto &blocks = iEvent.get(inputTagBlocks_);
  const auto &bid_name = output_names_[0];
  const auto &id_name = output_names_[1];
  const auto &momentum_name = output_names_[2];
  const auto &output1 = iOutput.at(bid_name);
  const auto &output2 = iOutput.at(id_name);
  const auto &output3 = iOutput.at(momentum_name);
  const auto &output_binary = output1.fromServer<float>();
  const auto &output_pid = output2.fromServer<float>();
  const auto &output_p4 = output3.fromServer<float>();
  auto candidates = makeCandidates(inputs,
                                   output_binary[0],
                                   output_pid[0],
                                   output_p4[0],
                                   blocks,
                                   additionalTrackFilterNoHCAL_,
                                   additionalTrackFilterParams_,
                                   noRegressionEndcap_,
                                   endcapTk_,
                                   selected_elements_);
  iEvent.emplace(pfCandidatesPutToken_, candidates);
}

void MLPFSONICProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  TritonClient::fillPSetDescription(desc);
  desc.add<edm::InputTag>("src", edm::InputTag("particleFlowBlock"));
  desc.add<bool>("additionalTrackFilterNoHCAL", false)
      ->setComment("Apply tight requirments for tracks not linked to HCAL clusters.");
  {
    edm::ParameterSetDescription psd;
    psd.add<std::string>("additionalFilterTk", "1.48 < abs(eta) < 3.0");
    psd.add<double>("pt_Error", 1.0)->setComment("Absolute pt error to detect fake tracks.");
    desc.add<edm::ParameterSetDescription>("AdditionalTrackFilterNoHCALParams", psd);
  }
  desc.add<bool>("noRegressionEndcap", false)->setComment("Turn on/off regression for tracks in the endcap.");
  desc.add<std::string>("endcapTk", "1.48 < abs(eta) < 3.0");
  desc.add<std::vector<std::string>>("input_names", {"mask", "Xfeat_normed"});
  desc.add<std::vector<std::string>>("output_names", {"bid", "id", "momentum"});
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MLPFSONICProducer);
