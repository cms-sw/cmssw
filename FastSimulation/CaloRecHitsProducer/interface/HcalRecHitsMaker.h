#ifndef FastSimulation__HcalRecHitsMaker__h
#define FastSimulation__HcalRecHitsMaker__h

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FastSimulation/Utilities/interface/GaussianTail.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <map>
#include <vector>

class CaloGeometry;
class RandomEngineAndDistribution;
class HcalSimParameterMap;
class HcalDbService;
class HcalRespCorrs;
class HcalTopology;

namespace edm { 
  class Event;
  class EventSetup;
  class ParameterSet;
}

class HcalRecHitsMaker
{
 public:
  HcalRecHitsMaker(edm::ParameterSet const & p,int);
  ~HcalRecHitsMaker();

  void loadHcalRecHits(edm::Event &iEvent, const HcalTopology&, HBHERecHitCollection& hbheHits, HBHEDigiCollection& hbheDigis,
                       RandomEngineAndDistribution const*);
  void loadHcalRecHits(edm::Event &iEvent, const HcalTopology&, HORecHitCollection &ho, HODigiCollection & hoDigis,
                       RandomEngineAndDistribution const*);
  void loadHcalRecHits(edm::Event &iEvent, const HcalTopology&, HFRecHitCollection &hfHits, HFDigiCollection& hfDigis,
                       RandomEngineAndDistribution const*);
  void init(const edm::EventSetup &es,bool dodigis,bool domiscalib);

 private:
  unsigned createVectorsOfCells(const edm::EventSetup &es);
  unsigned createVectorOfSubdetectorCells( const CaloGeometry&,const HcalTopology&, int subdetn,std::vector<int>&);
  unsigned noisifySubdet(std::vector<float >& theMap, std::vector<int>& theHits,const std::vector<int>& thecells, unsigned ncells, double  hcalHotFraction_, const GaussianTail *,double sigma,double threshold,double correctionfactor,
                         RandomEngineAndDistribution const*);
  // Not currently used. Will probably be removed soon.
  //  void noisifySignal(std::map<uint32_t,std::pair<float,bool> >& theMap); 
  void noisify(RandomEngineAndDistribution const*);
  double noiseInfCfromDB(const HcalDbService * conditions,const HcalDetId & detId);
  void Fill(int id,float energy, std::vector<int> & myHits,float noise,float correctionfactor, RandomEngineAndDistribution const*);
  void loadPCaloHits(const edm::Event & iEvent, const HcalTopology&, RandomEngineAndDistribution const*);
  
  void clean();
  void cleanSubDet(std::vector<float>& hits,std::vector<int>& cells);
  // conversion for digitization
  int fCtoAdc(double fc) const;
  double fractionOOT(int time_slice);

 private:
  unsigned det_;
  std::vector<double> threshold_;
  std::vector<double> noise_;
  std::vector<double> corrfac_;
  std::vector<double> hcalHotFraction_;
  unsigned nnoise_;

  //  edm::ESHandle<CaloTowerConstituentsMap> calotowerMap_;
  edm::InputTag inputCol_;
  bool initialized_;
  bool initializedHB_;
  bool initializedHE_;
  bool initializedHO_;
  bool initializedHF_;
  bool doDigis_;
  bool doMiscalib_;
  bool doSaturation_;
  std::vector<bool> noiseFromDb_;
  double refactor_;
  double refactor_mean_;
  std::string hcalfileinpath_;

  std::vector<float> hcalRecHits_;

  std::vector<int> firedCells_;

  std::vector<HcalDetId> theDetIds_;
  std::vector<float> miscalib_;

  // coefficients for fC to ADC conversion
  std::vector<int> fctoadc_;

  std::vector<float> peds_;
  std::vector<float> gains_;
  std::vector<float> sat_;
  std::vector<float> noisesigma_;
  std::vector<float> TPGFactor_;
 
  // the hashed indices
  unsigned maxIndex_;
  std::vector<int> hbhi_;
  std::vector<int> hehi_;
  std::vector<int> hohi_;
  std::vector<int> hfhi_;
  unsigned nhbcells_,nhecells_,nhocells_,nhfcells_;

  std::vector<GaussianTail*> myGaussianTailGenerators_;

  //  const HcalTPGCoder * myCoder_;
  //  HcalSimParameterMap * myHcalSimParameterMap_;

  // the access to the response corection factors
  const HcalRespCorrs* myRespCorr;
};

#endif
