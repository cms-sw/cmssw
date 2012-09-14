#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include <iostream>

class TestCaloSelectors : public edm::EDAnalyzer {
public:
  TestCaloSelectors(const edm::ParameterSet& ps) :
    inputTag_(ps.getParameter<edm::InputTag>("inputTag")) {
  }
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& es);
private:
  edm::InputTag inputTag_;
};

void TestCaloSelectors::analyze(const edm::Event& evt, const edm::EventSetup& c) {
  edm::Handle<HBHERecHitCollection> hbhe;
  evt.getByLabel(inputTag_, hbhe);
  HBHERecHitMetaCollection mhbhe(*hbhe);
  edm::ESHandle<CaloGeometry> pG;
  c.get<CaloGeometryRecord>().get(pG);

  double maxEt=-1;
  GlobalPoint pMax;
  for (CaloRecHitMetaCollectionV::const_iterator i=mhbhe.begin(); i!=mhbhe.end(); i++) {
    GlobalPoint p=pG->getPosition(i->detid());
    double et=i->energy()/cosh(p.eta());
    if (et>maxEt) {
      pMax=p;
      maxEt=et;
    }
  }
  

  CaloConeSelector sel(0.3, pG.product(), DetId::Hcal);
  CaloDualConeSelector sel2(0.3, 0.5, pG.product(), DetId::Hcal);
  
  std::auto_ptr<CaloRecHitMetaCollectionV> chosen=sel.select(pMax,mhbhe);
  std::auto_ptr<CaloRecHitMetaCollectionV> chosen2=sel2.select(pMax,mhbhe);

  std::cout << "Center at " << pMax.eta() << "," << pMax.phi() << " (ET=" << maxEt << ") I had " << mhbhe.size() << " and I kept " << chosen->size() << std::endl;
  
  for (CaloRecHitMetaCollectionV::const_iterator i=chosen->begin(); i!=chosen->end(); i++) {
    std::cout << HcalDetId(i->detid()) << " : " << (*i) << std::endl;
  }
  std::cout << "Dual cone\n";
  for (CaloRecHitMetaCollectionV::const_iterator i=chosen2->begin(); i!=chosen2->end(); i++) {
    std::cout << HcalDetId(i->detid()) << " : " << (*i) << std::endl;
  }

  std::cout << std::endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_FWK_MODULE(TestCaloSelectors);

