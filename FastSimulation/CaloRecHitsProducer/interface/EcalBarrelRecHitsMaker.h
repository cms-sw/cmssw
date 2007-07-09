#ifndef FastSimulation__EcalBarrelRecHitsMaker__h
#define FastSimulation__EcalBarrelRecHitsMaker__h

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
//#include <boost/cstdint.hpp>

class RandomEngine;

namespace edm { 
  class ParameterSet;
  class Event;  
  class EventSetup;
}

class EcalBarrelRecHitsMaker
{
 public:
  EcalBarrelRecHitsMaker(edm::ParameterSet const & p, edm::ParameterSet const & p2,const RandomEngine* );
  ~EcalBarrelRecHitsMaker();

  void loadEcalBarrelRecHits(edm::Event &iEvent, EBRecHitCollection & ecalHits,EBDigiCollection & ecaldigis);
  void init(const edm::EventSetup &es,bool dodigis,bool doMiscalib);

 private:
  void clean();
  void loadPCaloHits(const edm::Event & iEvent);
  void geVtoGainAdc(float e,unsigned& gain,unsigned &adc) const;
  
 private:
  bool doDigis_;
  bool doMisCalib_;
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  double calibfactor_;
  const RandomEngine* random_;
  bool noisified_;

  // array (size = 62000) of the energy in the barrel
  std::vector<float> theCalorimeterHits_;
  // array of the hashedindices in the previous array of the cells that received a hit
  std::vector<int> theFiredCells_;
  
  // equivalent of the EcalIntercalibConstants from RecoLocalCalo/EcalRecProducers/src/EcalRecHitProduer.cc
  std::vector<float> theCalibConstants_;

  //array conversion hashedIndex rawId
  std::vector<uint32_t> barrelRawId_;
  float adcToGeV_;
  float geVToAdc1_,geVToAdc2_,geVToAdc3_;
  unsigned minAdc_;
  unsigned maxAdc_;
  float t1_,t2_,sat_;
};

#endif
