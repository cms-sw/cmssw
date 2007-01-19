#ifndef FastSimulation__HcalRecHitsMaker__h
#define FastSimulation__HcalRecHitsMaker__h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"

#include "FastSimulation/Utilities/interface/GaussianTail.h"
#include "FastSimulation/CaloRecHitsProducer/interface/SignalHit.h"

#include <map>
#include <vector>

#include <boost/cstdint.hpp>

class CaloGeometry;
class RandomEngine;

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
  void noisifySubdet(std::map<SignalHit,float>& theMap, const std::vector<uint32_t>& thecells, unsigned ncells); 
  void noisify();
  void noisifyAndFill(uint32_t id,float energy, std::map<SignalHit,float>& myHits);
  void loadPSimHits(const edm::Event & iEvent);
  
  void clean();

 private:
  double threshold_;
  double noise_;
  double hcalHotFraction_; 

  //  edm::ESHandle<CaloTowerConstituentsMap> calotowerMap_;
  bool initialized_;
  std::map<SignalHit,float> hbheRecHits_;
  std::map<SignalHit,float> hoRecHits_;
  std::map<SignalHit,float> hfRecHits_;

  std::vector<uint32_t> hbhecells_;
  std::vector<uint32_t> hocells_;
  std::vector<uint32_t> hfcells_;
  unsigned nhbhecells_;
  unsigned nhocells_;
  unsigned nhfcells_;

  GaussianTail myGaussianTailGenerator_;
  const RandomEngine* random_;
};

#endif
