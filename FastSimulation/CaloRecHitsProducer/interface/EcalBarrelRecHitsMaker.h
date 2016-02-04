#ifndef FastSimulation__EcalBarrelRecHitsMaker__h
#define FastSimulation__EcalBarrelRecHitsMaker__h

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "FWCore/Utilities/interface/InputTag.h"
//#include <boost/cstdint.hpp>

class RandomEngine;
class EcalTrigTowerConstituentsMap;
class GaussianTail;

namespace edm { 
  class ParameterSet;
  class Event;  
  class EventSetup;
}

class EcalBarrelRecHitsMaker
{
 public:
  EcalBarrelRecHitsMaker(edm::ParameterSet const & p, const RandomEngine* );
  ~EcalBarrelRecHitsMaker();

  void loadEcalBarrelRecHits(edm::Event &iEvent, EBRecHitCollection & ecalHits,EBDigiCollection & ecaldigis);
  void init(const edm::EventSetup &es,bool dodigis,bool doMiscalib);

 private:
  void clean();
  void loadPCaloHits(const edm::Event & iEvent);
  void geVtoGainAdc(float e,unsigned& gain,unsigned &adc) const;
  void noisifyTriggerTowers();
  bool noisifyTriggerTower(unsigned tthi);
  void randomNoisifier();
  bool isHighInterest(int tthi);


 private:
  bool doDigis_;
  bool doMisCalib_;
  double refactor_;
  double refactor_mean_;
  // poor-man Selective Readout
  double threshold_;
  double noise_;
  double calibfactor_;
  double EBHotFraction_ ;
  const RandomEngine* random_;
  const GaussianTail* myGaussianTailGenerator_;

  bool noisified_;
  edm::InputTag inputCol_;
  // array (size = 62000) of the energy in the barrel
  std::vector<float> theCalorimeterHits_;
  // array of the hashedindices in the previous array of the cells that received a hit
  std::vector<int> theFiredCells_;
  // array of the hashedindices in the previous array of the cells that have a high noise fluctuation
  std::vector<int> applyZSCells_;  
  // equivalent of the EcalIntercalibConstants from RecoLocalCalo/EcalRecProducers/src/EcalRecHitProduer.cc
  std::vector<float> theCalibConstants_;

  //array conversion hashedIndex rawId
  std::vector<uint32_t> barrelRawId_;
  float adcToGeV_;
  float geVToAdc1_,geVToAdc2_,geVToAdc3_;
  unsigned minAdc_;
  unsigned maxAdc_;
  float t1_,t2_,sat_;

  const EcalTrigTowerConstituentsMap* eTTmap_;  
  // Array of the DetIds
  std::vector<EcalTrigTowerDetId> theTTDetIds_;
  // Transverse Energy of the TT
  std::vector<float> TTTEnergy_;
  // shot TTs
  std::vector<unsigned> theFiredTTs_;
  // treated TTs
  std::vector<bool> treatedTTs_;
  // neighboring TT DetIds
  std::vector<std::vector<int> > neighboringTTs_;
  // the crystals in a given TT 
  std::vector<std::vector<int> > crystalsinTT_;
  // the towers which have been looked at 
  std::vector<int> theTTofHighInterest_;
  // the status of the towers. A tower is of high interest if it or one of its neighbour is above the threshold
  std::vector<int> TTHighInterest_;
  // vector of parameter for custom noise simulation (size =4 : 0 & 1 define the gaussian shape
  // of the noise ; 2 & 3 define the sigma and threshold in ADC counts of the *OFFLINE* amplitude
 	  	 
  std::vector<double> highNoiseParameters_ ;
  bool doCustomHighNoise_;

  // selective readout threshold
  float SRThreshold_;
  int SREtaSize_;
  int SRPhiSize_;
  // theta of the ieta 
  std::vector<float> sinTheta_;
  // the cell-dependant noise sigma 
  std::vector<float> noisesigma_;
  double meanNoiseSigmaEt_ ;
  // noise in ADC counts
  double noiseADC_;
  // need to keep the address of ICMC
  const std::vector<float> * ICMC_;

};

#endif
