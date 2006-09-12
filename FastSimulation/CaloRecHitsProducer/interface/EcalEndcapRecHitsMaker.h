#ifndef FastSimulation__EcalEndcapRecHitsMaker__h
#define FastSimulation__EcalEndcapRecHitsMaker__h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"

class EcalEndcapRecHitsMaker
{
 public:
  EcalEndcapRecHitsMaker(edm::ParameterSet const & p);
  ~EcalEndcapRecHitsMaker();

  void loadEcalEndcapRecHits(edm::Event &iEvent, EERecHitCollection & ecalHits);
    
 private:
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  
};

#endif
