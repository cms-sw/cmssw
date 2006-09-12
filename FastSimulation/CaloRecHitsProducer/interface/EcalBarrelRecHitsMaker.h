#ifndef FastSimulation__EcalBarrelRecHitsMaker__h
#define FastSimulation__EcalBarrelRecHitsMaker__h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"

class EcalBarrelRecHitsMaker
{
 public:
  EcalBarrelRecHitsMaker(edm::ParameterSet const & p);
  ~EcalBarrelRecHitsMaker();

  void loadEcalBarrelRecHits(edm::Event &iEvent, EBRecHitCollection & ecalHits);
    
 private:
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  
};

#endif
