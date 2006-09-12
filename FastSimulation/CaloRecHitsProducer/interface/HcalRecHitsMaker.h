#ifndef FastSimulation__HcalRecHitsMaker__h
#define FastSimulation__HcalRecHitsMaker__h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/ESHandle.h"
//Test
//#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
// End of test


class HcalRecHitsMaker
{
 public:
  HcalRecHitsMaker(edm::ParameterSet const & p);
  ~HcalRecHitsMaker();

  void loadHcalRecHits(edm::Event &iEvent, HBHERecHitCollection& hbheHits, HORecHitCollection &hoHits,HFRecHitCollection &hfHits);
  void init(const edm::EventSetup &es);

 private:
  double threshold_;
  double noise_;
  //  edm::ESHandle<CaloTowerConstituentsMap> calotowerMap_;
  bool initialized_;
};

#endif
