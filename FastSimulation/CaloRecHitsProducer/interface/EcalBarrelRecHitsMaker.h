#ifndef FastSimulation__EcalBarrelRecHitsMaker__h
#define FastSimulation__EcalBarrelRecHitsMaker__h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include <map>
#include <boost/cstdint.hpp>

class RandomEngine;

class EcalBarrelRecHitsMaker
{
 public:
  EcalBarrelRecHitsMaker(edm::ParameterSet const & p, const RandomEngine* );
  ~EcalBarrelRecHitsMaker();

  void loadEcalBarrelRecHits(edm::Event &iEvent, EBRecHitCollection & ecalHits);

 private:
  void clean();
  void loadPSimHits(const edm::Event & iEvent);
  void noisifyAndFill(uint32_t id,float energy, std::map<uint32_t,float>& myHits);
    
 private:
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  std::map<uint32_t,float> ecalbRecHits_;
  const RandomEngine* random_;
};

#endif
