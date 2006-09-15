#ifndef FastSimulation__EcalBarrelRecHitsMaker__h
#define FastSimulation__EcalBarrelRecHitsMaker__h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include <vector>

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
  std::vector<float> barrelhits_;
  std::vector<bool> saved_;
  std::vector<int> hittosave_;
};

#endif
