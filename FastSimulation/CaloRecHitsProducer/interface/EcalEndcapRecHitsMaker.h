#ifndef FastSimulation__EcalEndcapRecHitsMaker__h
#define FastSimulation__EcalEndcapRecHitsMaker__h

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
//#include <boost/cstdint.hpp>

class RandomEngine;
class EcalTrigTowerConstituentsMap;
class GaussianTail;

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

  void loadEcalEndcapRecHits(edm::Event &iEvent, EERecHitCollection & ecalHits,EEDigiCollection & ecalDigis);
  void init(const edm::EventSetup &es,bool dodigis,bool domiscalib);

 private:
  void clean();
  void loadPCaloHits(const edm::Event & iEvent);
  void geVtoGainAdc(float e,unsigned & gain, unsigned &adc) const;
  // there are 2448 TT in the barrel. 
  inline int TThashedIndexforEE(int originalhi) const {return originalhi-2448;}
  inline int TThashedIndexforEE(const EcalTrigTowerDetId &detid) const {return detid.hashedIndex()-2448;}
  // the number of the SuperCrystals goes from 1 to 316 (with some holes) in each EE
  // z should -1 or 1 
  inline  int SChashedIndex(int SC,int z) const {return SC+(z+1)*158;}
  inline int SChashedIndex(const EEDetId& detid) const {
    //    std::cout << "In SC hashedIndex " <<  detid.isc() << " " << detid.zside() << " " << detid.isc()+(detid.zside()+1)*158 << std::endl;
    return detid.isc()+(detid.zside()+1)*158;}
  inline int towerOf(const EEDetId& detid) const {return towerOf_[detid.hashedIndex()];}
  inline int towerOf(int hid) const {return towerOf_[hid];}
  void noisifyTriggerTowers();
  void noisifySuperCrystals(int tthi);
  void randomNoisifier();
  bool isHighInterest(const EEDetId & icell);

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
  double EEHotFraction_ ;
  const RandomEngine* random_;
  const GaussianTail * myGaussianTailGenerator_;
  bool noisified_;

  // array (size = 20000) of the energy in the barrel
  std::vector<float> theCalorimeterHits_;
  // array of the hashedindices in the previous array of the cells that received a hit
  std::vector<int> theFiredCells_;
  // array of the hashedindices in the previous array of the cells that received a hit
  std::vector<int> applyZSCells_;

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

  const EcalTrigTowerConstituentsMap* eTTmap_;  

  // arraws for the selective readout emulation
  // fast EEDetId -> TT hashedindex conversion
  std::vector<int>  towerOf_;
  // vector of the original DetId if needed
  std::vector<EcalTrigTowerDetId> theTTDetIds_;
  // list of SC "contained" in a TT.
  std::vector<std::vector<int> > SCofTT_;
  // list of TT of a given sc
  std::vector<std::vector<int> > TTofSC_;
  // status of each SC 
  std::vector<bool> treatedSC_;
  std::vector<int> SCHighInterest_;
  // list of fired SC
  std::vector<int> theFiredSC_;
  // the list of fired TT
  std::vector<int> theFiredTTs_;
  // the energy in the TT
  std::vector<float> TTTEnergy_;
  // the cells in each SC
  std::vector<std::vector<int> > CrystalsinSC_;
  // the sin(theta) of the cell
  std::vector<float> sinTheta_;

  // the cell-dependant noise sigma 
  std::vector<float> noisesigma_;
  double meanNoiseSigmaEt_ ;
  // noise in ADC counts
  double noiseADC_;
  // selective readout threshold
  float SRThreshold_;
  // need to keep the address of ICMC
  const std::vector<float> * ICMC_;
  // vector of parameter for custom noise simulation (size =4 : 0 & 1 define the gaussian shape 
  // of the noise ; 2 & 3 define the sigma and threshold in ADC counts of the *OFFLINE* amplitude

  std::vector<double> highNoiseParameters_ ; 
  bool doCustomHighNoise_;
};

#endif
