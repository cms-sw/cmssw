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
  void init(const edm::EventSetup &es);

 private:
  void clean();
  void loadPCaloHits(const edm::Event & iEvent);
  void noisifyAndFill(uint32_t id,float energy, std::map<uint32_t,float>& myHits);
    
 private:
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  std::map<uint32_t,float> ecalbRecHits_;
  const RandomEngine* random_;
  bool noisified_;

  // array (size = 62000) of the energy in the barrel
  std::vector<float> theCalorimeterHits_;
  // array of the hashedindices in the previous array of the cells that received a hit
  std::vector<int> theFiredCells_;

  //array conversion hashedIndex rawId
  std::vector<uint32_t> barrelRawId_;
};

#endif
