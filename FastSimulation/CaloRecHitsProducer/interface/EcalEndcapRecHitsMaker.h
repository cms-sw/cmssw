#ifndef FastSimulation__EcalEndcapRecHitsMaker__h
#define FastSimulation__EcalEndcapRecHitsMaker__h

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
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
  EcalEndcapRecHitsMaker(edm::ParameterSet const & p,edm::ParameterSet const & p2,const RandomEngine* random);
  ~EcalEndcapRecHitsMaker();

  void loadEcalEndcapRecHits(edm::Event &iEvent, EERecHitCollection & ecalHits,EEDigiCollection & ecalDigis);
  void init(const edm::EventSetup &es,bool dodigis,bool domiscalib);

 private:
  void clean();
  void loadPCaloHits(const edm::Event & iEvent);
  void geVtoGainAdc(float e,unsigned & gain, unsigned &adc) const;

 private:
  edm::InputTag inputCol_;
  bool doDigis_;
  bool doMisCalib_;
  double refactor_;
  double refactor_mean_;
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  double calibfactor_;

  const RandomEngine* random_;
  bool noisified_;

  // array (size = 20000) of the energy in the barrel
  std::vector<float> theCalorimeterHits_;
  // array of the hashedindices in the previous array of the cells that received a hit
  std::vector<int> theFiredCells_;

  // equivalent of the EcalIntercalibConstants from RecoLocalCalo/EcalRecProducers/src/EcalRecHitProduer.cc
  std::vector<float> theCalibConstants_;

  //array conversion hashedIndex rawId
  std::vector<uint32_t> endcapRawId_;

  
  // digitization
  float adcToGeV_;
  float geVToAdc1_,geVToAdc2_,geVToAdc3_;
  unsigned minAdc_;
  unsigned maxAdc_;
  float t1_,t2_,sat_;
};

#endif
