#ifndef FastSimulation__EcalEndcapRecHitsMaker__h
#define FastSimulation__EcalEndcapRecHitsMaker__h

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include <map>
//#include <boost/cstdint.hpp>

class RandomEngine;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class EcalEndcapRecHitsMaker
{
 public:
  EcalEndcapRecHitsMaker(edm::ParameterSet const & p,const RandomEngine* random);
  ~EcalEndcapRecHitsMaker();

  void loadEcalEndcapRecHits(edm::Event &iEvent, EERecHitCollection & ecalHits);
  void init(const edm::EventSetup &es);

 private:
  void clean();
  void loadPCaloHits(const edm::Event & iEvent);
  void noisifyAndFill(uint32_t id,float energy, std::map<uint32_t,float>& myHits);
    
 private:
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  std::map<uint32_t,float> ecaleRecHits_;

  const RandomEngine* random_;
  bool noisified_;

  // array (size = 20000) of the energy in the barrel
  std::vector<float> theCalorimeterHits_;
  // array of the hashedindices in the previous array of the cells that received a hit
  std::vector<int> theFiredCells_;

  //array conversion hashedIndex rawId
  std::vector<uint32_t> endcapRawId_;
};

#endif
