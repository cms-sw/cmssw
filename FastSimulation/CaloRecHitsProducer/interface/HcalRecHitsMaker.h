#ifndef FastSimulation__HcalRecHitsMaker__h
#define FastSimulation__HcalRecHitsMaker__h

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "FastSimulation/Utilities/interface/GaussianTail.h"

#include <map>
#include <vector>

//#include <boost/cstdint.hpp>

class CaloGeometry;
class RandomEngine;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class HcalRecHitsMaker
{
 public:
  HcalRecHitsMaker(edm::ParameterSet const & p, const RandomEngine* random);
  ~HcalRecHitsMaker();

  void loadHcalRecHits(edm::Event &iEvent, HBHERecHitCollection& hbheHits, HORecHitCollection &hoHits,HFRecHitCollection &hfHits);
  void init(const edm::EventSetup &es);

 private:
  unsigned createVectorsOfCells(const edm::EventSetup &es);
  unsigned createVectorOfSubdetectorCells( const CaloGeometry&,int subdetn,std::vector<uint32_t>&);
  void noisifySubdet(std::map<uint32_t,std::pair<float,bool> >& theMap, const std::vector<uint32_t>& thecells, unsigned ncells, double  hcalHotFraction_); 
  // Not currently used. Will probably be removed soon.
  //  void noisifySignal(std::map<uint32_t,std::pair<float,bool> >& theMap); 
  void noisify();
  void Fill(uint32_t id,float energy, std::map<uint32_t,std::pair<float,bool> >& myHits,bool signal=true, double noise_=0.);
  void loadPCaloHits(const edm::Event & iEvent);
  
  void clean();

 private:
  double thresholdHB_,  thresholdHE_, thresholdHO_, thresholdHF_;
  double noiseHB_, noiseHE_, noiseHO_, noiseHF_;
  double hcalHotFractionHB_,  hcalHotFractionHE_, hcalHotFractionHO_, hcalHotFractionHF_; 

  //  edm::ESHandle<CaloTowerConstituentsMap> calotowerMap_;
  
  bool initialized_;
  //the bool means killed ! 
  std::map<uint32_t,std::pair<float,bool> > hbRecHits_;
  std::map<uint32_t,std::pair<float,bool> > heRecHits_;
  std::map<uint32_t,std::pair<float,bool> > hoRecHits_;
  std::map<uint32_t,std::pair<float,bool> > hfRecHits_;

  std::vector<uint32_t> hbcells_;
  std::vector<uint32_t> hecells_;
  std::vector<uint32_t> hocells_;
  std::vector<uint32_t> hfcells_;
  unsigned nhbcells_;
  unsigned nhecells_;
  unsigned nhocells_;
  unsigned nhfcells_;

  const RandomEngine* random_;
  const GaussianTail* myGaussianTailGenerator_;

};

#endif
