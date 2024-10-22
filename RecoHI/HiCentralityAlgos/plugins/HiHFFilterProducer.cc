
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/HeavyIonEvent/interface/HFFilterInfo.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"

#include "DataFormats/Common/interface/Wrapper.h"

#include <cmath>

class HiHFFilterProducer : public edm::stream::EDProducer<> {
public:
  explicit HiHFFilterProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<CaloTowerCollection> srcTowers_;
};

HiHFFilterProducer::HiHFFilterProducer(const edm::ParameterSet& iConfig)
    : srcTowers_(consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("srcTowers"))) {
  produces<reco::HFFilterInfo>("hiHFfilters");
}

void HiHFFilterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;
  using namespace edm;

  auto const& towers = iEvent.get(srcTowers_);

  unsigned short int nTowersTh2HFplus = 0;
  unsigned short int nTowersTh2HFminus = 0;
  unsigned short int nTowersTh3HFplus = 0;
  unsigned short int nTowersTh3HFminus = 0;
  unsigned short int nTowersTh4HFplus = 0;
  unsigned short int nTowersTh4HFminus = 0;
  unsigned short int nTowersTh5HFplus = 0;
  unsigned short int nTowersTh5HFminus = 0;

  auto hiHFFilterResults = std::make_unique<reco::HFFilterInfo>();
  for (const auto& tower : towers) {
    const auto et = tower.et();
    const auto energy = tower.energy();
    const auto eta = tower.eta();
    const bool eta_plus = (eta > 3.0) && (eta < 6.0);
    const bool eta_minus = (eta < -3.0) && (eta > -6.0);
    if (et < 0.0)
      continue;
    if (eta_plus) {
      nTowersTh2HFplus += energy >= 2.0 ? 1 : 0;
      nTowersTh3HFplus += energy >= 3.0 ? 1 : 0;
      nTowersTh4HFplus += energy >= 4.0 ? 1 : 0;
      nTowersTh5HFplus += energy >= 5.0 ? 1 : 0;
    } else if (eta_minus) {
      nTowersTh2HFminus += energy >= 2.0 ? 1 : 0;
      nTowersTh3HFminus += energy >= 3.0 ? 1 : 0;
      nTowersTh4HFminus += energy >= 4.0 ? 1 : 0;
      nTowersTh5HFminus += energy >= 5.0 ? 1 : 0;
    }
  }

  hiHFFilterResults->numMinHFTowers2 = std::min(nTowersTh2HFplus, nTowersTh2HFminus);
  hiHFFilterResults->numMinHFTowers3 = std::min(nTowersTh3HFplus, nTowersTh3HFminus);
  hiHFFilterResults->numMinHFTowers4 = std::min(nTowersTh4HFplus, nTowersTh4HFminus);
  hiHFFilterResults->numMinHFTowers5 = std::min(nTowersTh5HFplus, nTowersTh5HFminus);

  iEvent.put(std::move(hiHFFilterResults), "hiHFfilters");
}

void HiHFFilterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcTowers", {"towerMaker"});
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HiHFFilterProducer);
