#ifndef FastSimulation__EcalEndcapRecHitsMaker__h
#define FastSimulation__EcalEndcapRecHitsMaker__h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include <map>
#include <boost/cstdint.hpp>

class RandomEngine;

class EcalEndcapRecHitsMaker
{
 public:
  EcalEndcapRecHitsMaker(edm::ParameterSet const & p,const RandomEngine* random);
  ~EcalEndcapRecHitsMaker();

  void loadEcalEndcapRecHits(edm::Event &iEvent, EERecHitCollection & ecalHits);

 private:
  void clean();
  void loadPSimHits(const edm::Event & iEvent);
  void noisifyAndFill(uint32_t id,float energy, std::map<uint32_t,float>& myHits);
    
 private:
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  std::map<uint32_t,float> ecaleRecHits_;

  const RandomEngine* random_;
};

#endif
