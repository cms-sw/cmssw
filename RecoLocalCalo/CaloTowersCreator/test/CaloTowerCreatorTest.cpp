#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerHardcodeGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <iostream>

int main() {

  HcalTopology topology;
  HcalHardcodeGeometryLoader loader(topology);
  std::auto_ptr<CaloSubdetectorGeometry> hcalGeometry = loader.load();

  CaloGeometry geometry;
  geometry.setSubdetGeometry(DetId::Hcal, HcalBarrel, hcalGeometry.get());
  geometry.setSubdetGeometry(DetId::Hcal, HcalEndcap, hcalGeometry.get());
  geometry.setSubdetGeometry(DetId::Hcal, HcalOuter,  hcalGeometry.get());
  geometry.setSubdetGeometry(DetId::Hcal, HcalForward, hcalGeometry.get());

  CaloTowerHardcodeGeometryLoader towerLoader;
  std::auto_ptr<CaloSubdetectorGeometry> towerGeometry = towerLoader.load();
  
  geometry.setSubdetGeometry(DetId::Calo, 1, towerGeometry.get()); 

  CaloTowersCreationAlgo algo;
  algo.setGeometry(&topology, &geometry);

  // make one RecHit, with energy 1 GeV, for
  // every cell with a given phi
  int magicPhi = 5;

  HBHERecHitCollection hbheHits;
  HORecHitCollection hoHits;
  HFRecHitCollection hfHits;

  std::vector<DetId>::const_iterator detItr;

  std::vector<DetId> hbDets = geometry.getValidDetIds(DetId::Hcal, HcalBarrel);
  for(detItr = hbDets.begin(); detItr != hbDets.end(); ++detItr) {
    if(HcalDetId(*detItr).iphi() == magicPhi) {
      hbheHits.push_back(HBHERecHit(*detItr, 1., 0));
    }
  }

  std::vector<DetId> heDets = geometry.getValidDetIds(DetId::Hcal, HcalEndcap);
  for(detItr = heDets.begin(); detItr != heDets.end(); ++detItr) {
    if(HcalDetId(*detItr).iphi() == magicPhi) {
      hbheHits.push_back(HBHERecHit(*detItr, 1., 0));
    }
  }

  std::vector<DetId> hoDets = geometry.getValidDetIds(DetId::Hcal, HcalOuter);
  for(detItr = hoDets.begin(); detItr != hoDets.end(); ++detItr) {
    if(HcalDetId(*detItr).iphi() == magicPhi) {
      hoHits.push_back(HORecHit(*detItr, 1., 0));
    }
  }

  std::vector<DetId> hfDets = geometry.getValidDetIds(DetId::Hcal, HcalForward);
  for(detItr = hfDets.begin(); detItr != hfDets.end(); ++detItr) {
    if(HcalDetId(*detItr).iphi() == magicPhi) {
      hfHits.push_back(HFRecHit(*detItr, 1., 0));
    }
  }

std::cout << "NUMBER OF HITS " << hbheHits.size() << " " << hoHits.size() << " " << hfHits.size() << std::endl;
  // do the actual tower building
  CaloTowerCollection collection;

  algo.create(collection, hbheHits, hoHits, hfHits);


  for(CaloTowerCollection::const_iterator towerItr = collection.begin();
      towerItr != collection.end(); ++towerItr)
  {
//    std::cout << towerItr->id().ieta() << " " << towerItr->e() << std::endl;
  }

  // now make a messed-up one, which uses draconian thresholds on HED, and
  // silly weights on HF
  CaloTowersCreationAlgo sillyAlgo(0.,0.,0., 0., 0., 10000.,
                                   0.,0.,0.,
                                   1.,1.,1.,1.,1.,
                                   1., 1., 1.1,
                                    0.,0.,0., true); 
  sillyAlgo.setGeometry(&topology, &geometry);
  
  CaloTowerCollection collection2;
  sillyAlgo.create(collection2, hbheHits, hoHits, hfHits);
  for(CaloTowerCollection::const_iterator towerItr = collection2.begin();
      towerItr != collection2.end(); ++towerItr)
  {
    std::cout << towerItr->id().ieta() << " " << towerItr->e() << std::endl;
  }

}

