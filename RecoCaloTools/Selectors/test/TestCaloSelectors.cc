#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include <iostream>
#include <functional>

class TestCaloSelectors : public edm::one::EDAnalyzer<> {
public:
  TestCaloSelectors(const edm::ParameterSet& ps)
      : token_(consumes(ps.getParameter<edm::InputTag>("inputTag"))), caloToken_(esConsumes()) {}
  void analyze(const edm::Event& evt, const edm::EventSetup& es) override;

private:
  edm::EDGetTokenT<HBHERecHitCollection> token_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloToken_;
};

void TestCaloSelectors::analyze(const edm::Event& evt, const edm::EventSetup& c) {
  edm::ESHandle<CaloGeometry> pG = c.getHandle(caloToken_);

  const HBHERecHitCollection& mhbhe = evt.get(token_);

  double maxEt = -1;
  GlobalPoint pMax;
  for (HBHERecHitCollection::const_iterator i = mhbhe.begin(); i != mhbhe.end(); i++) {
    GlobalPoint p = pG->getPosition(i->detid());
    double et = i->energy() / cosh(p.eta());
    if (et > maxEt) {
      pMax = p;
      maxEt = et;
    }
  }

  CaloConeSelector<HBHERecHit> sel(0.3, pG.product(), DetId::Hcal);
  CaloDualConeSelector<HBHERecHit> sel2(0.3, 0.5, pG.product(), DetId::Hcal);

  std::cout << "Center at " << pMax.eta() << "," << pMax.phi() << " (ET=" << maxEt << ")" << std::endl;

  sel.selectCallback(
      pMax, mhbhe, [&](const HBHERecHit& i) { std::cout << HcalDetId(i.detid()) << " : " << i << std::endl; });

  std::cout << "Dual cone\n";

  sel2.selectCallback(
      pMax, mhbhe, [&](const HBHERecHit& i) { std::cout << HcalDetId(i.detid()) << " : " << i << std::endl; });

  std::cout << std::endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TestCaloSelectors);
