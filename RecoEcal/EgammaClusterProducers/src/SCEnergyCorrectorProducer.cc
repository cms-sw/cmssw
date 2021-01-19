#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEcal/EgammaClusterAlgos/interface/SCEnergyCorrectorSemiParm.h"

#include <vector>

//A simple producer which produces a set of corrected superclusters
//Note this is more for testing and development and is not really meant for production
//although its perfectly possible somebody could use it in some prod workflow
//author S. Harper (RAL/CERN)

class SCEnergyCorrectorProducer : public edm::stream::EDProducer<> {
public:
  explicit SCEnergyCorrectorProducer(const edm::ParameterSet& iConfig);

  void beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  SCEnergyCorrectorSemiParm energyCorrector_;
  edm::EDGetTokenT<reco::SuperClusterCollection> inputSCToken_;
  bool writeFeatures_;
};

SCEnergyCorrectorProducer::SCEnergyCorrectorProducer(const edm::ParameterSet& iConfig)
    : energyCorrector_(iConfig.getParameterSet("correctorCfg"), consumesCollector()),
      inputSCToken_(consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("inputSCs"))),
      writeFeatures_(iConfig.getParameter<bool>("writeFeatures")) {
  produces<reco::SuperClusterCollection>();
  if (writeFeatures_) {
    produces<std::vector<std::vector<float>>>("features");
  }
}

void SCEnergyCorrectorProducer::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) {
  energyCorrector_.setEventSetup(iSetup);
}

void SCEnergyCorrectorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  energyCorrector_.setEvent(iEvent);

  auto inputSCHandle = iEvent.getHandle(inputSCToken_);
  auto corrSCs = std::make_unique<reco::SuperClusterCollection>();
  auto scFeatures = std::make_unique<std::vector<std::vector<float>>>();
  for (const auto& inputSC : *inputSCHandle) {
    corrSCs->push_back(inputSC);
    energyCorrector_.modifyObject(corrSCs->back());
    if (writeFeatures_) {
      scFeatures->emplace_back(energyCorrector_.getRegData(corrSCs->back()));
    }
  }
  iEvent.put(std::move(corrSCs));
  iEvent.put(std::move(scFeatures), "features");
}
void SCEnergyCorrectorProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::ParameterSetDescription>("correctorCfg", SCEnergyCorrectorSemiParm::makePSetDescription());
  desc.add<bool>("writeFeatures", false);
  desc.add<edm::InputTag>("inputSCs", edm::InputTag("particleFlowSuperClusterECAL"));
  descriptions.add("scEnergyCorrectorProducer", desc);
}

DEFINE_FWK_MODULE(SCEnergyCorrectorProducer);
