
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowerCandidateCreator.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"

#include <cmath>

class HiHFFilterProducer : public edm::stream::EDProducer<> {
public:
  explicit HiHFFilterProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<CaloTowerCollection>  srcTowers_;

};

HiHFFilterProducer::HiHFFilterProducer(const edm::ParameterSet& iConfig)
    : srcTowers_(consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("srcTowers"))) {
  produces<std::vector<int>>("hiHFfilters");
}

void HiHFFilterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  auto const& towers = iEvent.getHandle(srcTowers_);

  int nTowersTh2HFplus = 0;
  int nTowersTh2HFminus = 0;
  int nTowersTh3HFplus = 0;
  int nTowersTh3HFminus = 0;
  int nTowersTh4HFplus = 0;
  int nTowersTh4HFminus = 0;
  int nTowersTh5HFplus = 0;
  int nTowersTh5HFminus = 0;

  std::size_t size = 4;
  auto hiHFfiltersOut = std::make_unique<std::vector<int>>(size);

  for( unsigned int towerIndx = 0; towerIndx<towers->size(); ++towerIndx){
    const CaloTower & tower = (*towers)[ towerIndx ];
    const auto et = tower.et();
    const auto energy = tower.energy();
    const auto eta = tower.eta();
    const bool eta_plus = (eta > 3.0) && (eta < 6.0);
    const bool eta_minus = (eta < -3.0) && (eta > -6.0);
    if(et < 0.0)continue;
    if(eta_plus){
      if(energy >= 2.0)
        nTowersTh2HFplus += 1;
      if(energy >= 3.0)
        nTowersTh3HFplus += 1;
      if(energy >= 4.0)
        nTowersTh4HFplus += 1;
      if(energy >= 5.0)
        nTowersTh5HFplus += 1;
    }else if(eta_minus){
      if(energy >= 2.0)
        nTowersTh2HFminus += 1;
      if(energy >= 3.0)
        nTowersTh3HFminus += 1;
      if(energy >= 4.0)
        nTowersTh4HFminus += 1;
      if(energy >= 5.0)
        nTowersTh5HFminus += 1;
    }
  }

  (*hiHFfiltersOut)[0] = std::min(nTowersTh2HFplus, nTowersTh2HFminus);
  (*hiHFfiltersOut)[1] = std::min(nTowersTh3HFplus, nTowersTh3HFminus);
  (*hiHFfiltersOut)[2] = std::min(nTowersTh4HFplus, nTowersTh4HFminus);
  (*hiHFfiltersOut)[3] = std::min(nTowersTh5HFplus, nTowersTh5HFminus);

  iEvent.put(std::move(hiHFfiltersOut), "hiHFfilters");

}


void HiHFFilterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcTowers", {"towerMaker"});
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HiHFFilterProducer);
