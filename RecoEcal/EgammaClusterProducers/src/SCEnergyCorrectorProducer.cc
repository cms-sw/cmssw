#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
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
  const edm::EDGetTokenT<reco::SuperClusterCollection> inputSCToken_;
  const bool writeFeatures_;
};

SCEnergyCorrectorProducer::SCEnergyCorrectorProducer(const edm::ParameterSet& iConfig)
    : energyCorrector_(iConfig.getParameterSet("correctorCfg"), consumesCollector()),
      inputSCToken_(consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("inputSCs"))),
      writeFeatures_(iConfig.getParameter<bool>("writeFeatures")) {
  produces<reco::SuperClusterCollection>();
  if (writeFeatures_) {
    produces<edm::ValueMap<std::vector<float>>>("features");
  }
}

void SCEnergyCorrectorProducer::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) {
  energyCorrector_.setEventSetup(iSetup);
}

void SCEnergyCorrectorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  energyCorrector_.setEvent(iEvent);

  auto inputSCs = iEvent.get(inputSCToken_);
  auto corrSCs = std::make_unique<reco::SuperClusterCollection>();
  std::vector<std::vector<float>> scFeatures;
  for (const auto& inputSC : inputSCs) {
    corrSCs->push_back(inputSC);
    energyCorrector_.modifyObject(corrSCs->back());
    if (writeFeatures_) {
      scFeatures.emplace_back(energyCorrector_.getRegData(corrSCs->back()));
    }
  }

  auto scHandle = iEvent.put(std::move(corrSCs));

  if (writeFeatures_) {
    auto valMap = std::make_unique<edm::ValueMap<std::vector<float>>>();
    edm::ValueMap<std::vector<float>>::Filler filler(*valMap);
    filler.insert(scHandle, scFeatures.begin(), scFeatures.end());
    filler.fill();
    iEvent.put(std::move(valMap), "features");
  }
}

void SCEnergyCorrectorProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::ParameterSetDescription>("correctorCfg", SCEnergyCorrectorSemiParm::makePSetDescription());
  desc.add<bool>("writeFeatures", false);
  desc.add<edm::InputTag>("inputSCs", edm::InputTag("particleFlowSuperClusterECAL"));
  descriptions.add("scEnergyCorrectorProducer", desc);
}

DEFINE_FWK_MODULE(SCEnergyCorrectorProducer);
