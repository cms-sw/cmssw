#ifndef FastSimulation__HcalRecHitsMaker__h
#define FastSimulation__HcalRecHitsMaker__h

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FastSimulation/Utilities/interface/GaussianTail.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include <map>
#include <vector>

//#include <boost/cstdint.hpp>

class CaloGeometry;
class RandomEngine;
class HcalTPGCoder;
class HcalSimParameterMap;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class HcalRecHitsMaker
{
 public:
  HcalRecHitsMaker(edm::ParameterSet const & p, edm::ParameterSet const & p,const RandomEngine* random);
  ~HcalRecHitsMaker();

  void loadHcalRecHits(edm::Event &iEvent, HBHERecHitCollection& hbheHits, HORecHitCollection &hoHits,HFRecHitCollection &hfHits, HBHEDigiCollection& hbheDigis, HODigiCollection & hoDigis, HFDigiCollection& hfDigis);
  void init(const edm::EventSetup &es,bool dodigis,bool domiscalib);

 private:
  unsigned createVectorsOfCells(const edm::EventSetup &es);
  unsigned createVectorOfSubdetectorCells( const CaloGeometry&,int subdetn,std::vector<int>&);
  unsigned noisifySubdet(std::vector<float >& theMap, std::vector<int>& theHits,const std::vector<int>& thecells, unsigned ncells, double  hcalHotFraction_, const GaussianTail *,double sigma,double threshold); 
  // Not currently used. Will probably be removed soon.
  //  void noisifySignal(std::map<uint32_t,std::pair<float,bool> >& theMap); 
  void noisify();
  void Fill(int id,float energy, std::vector<int> & myHits,float noise);
  void loadPCaloHits(const edm::Event & iEvent);
  
  void clean();
  void cleanSubDet(std::vector<float>& hits,std::vector<int>& cells);
  // conversion for digitization
  int fCtoAdc(double fc) const;

 private:
  float thresholdHB_,  thresholdHE_, thresholdHO_, thresholdHF_;
  float  satHB_;
  float  satHE_;
  float  satHO_;
  float  satHF_;
  float noiseHB_, noiseHE_, noiseHO_, noiseHF_;
  double hcalHotFractionHB_,  hcalHotFractionHE_, hcalHotFractionHO_, hcalHotFractionHF_; 

  //  edm::ESHandle<CaloTowerConstituentsMap> calotowerMap_;
  edm::InputTag inputCol_;
  bool initialized_;
  bool doDigis_;
  bool doMiscalib_;
  double refactor_;
  double refactor_mean_;
  std::string hcalfileinpath_;

  std::vector<float> hcalRecHits_;

  std::vector<int> firedCellsHB_;
  std::vector<int> firedCellsHE_;
  std::vector<int> firedCellsHO_;
  std::vector<int> firedCellsHF_;

  std::vector<HcalDetId> theDetIds_;
  std::vector<float> miscalib_;

  // coefficients for fC to ADC conversion
  std::vector<int> fctoadc_;

  std::vector<float> peds_;
  std::vector<float> gains_;

  std::vector<float> TPGFactor_;
 
  // the hashed indices
  unsigned maxIndex_;
  unsigned maxIndexDebug_;
  std::vector<int> hbhi_;
  std::vector<int> hehi_;
  std::vector<int> hohi_;
  std::vector<int> hfhi_;
  unsigned nhbcells_;
  unsigned nhecells_;
  unsigned nhocells_;
  unsigned nhfcells_;

  const RandomEngine* random_;
  const GaussianTail* myGaussianTailGeneratorHB_;
  const GaussianTail* myGaussianTailGeneratorHE_;
  const GaussianTail* myGaussianTailGeneratorHO_;
  const GaussianTail* myGaussianTailGeneratorHF_;

  const HcalTPGCoder * myCoder_;
  HcalSimParameterMap * myHcalSimParameterMap_;
};

#endif
