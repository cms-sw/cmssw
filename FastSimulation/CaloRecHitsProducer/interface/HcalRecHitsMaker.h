#ifndef FastSimulation__HcalRecHitsMaker__h
#define FastSimulation__HcalRecHitsMaker__h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"

#include "FastSimulation/Utilities/interface/GaussianTail.h"

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
  void noisifySubdet(std::map<uint32_t,std::pair<float,bool> >& theMap, const std::vector<uint32_t>& thecells, unsigned ncells); 
  // Not currently used. Will probably be removed soon.
  //  void noisifySignal(std::map<uint32_t,std::pair<float,bool> >& theMap); 
  void noisify();
  void Fill(uint32_t id,float energy, std::map<uint32_t,std::pair<float,bool> >& myHits,bool signal=true);
  void loadPCaloHits(const edm::Event & iEvent);
  
  void clean();

 private:
  double threshold_;
  double noise_;
  double hcalHotFraction_; 

  //  edm::ESHandle<CaloTowerConstituentsMap> calotowerMap_;
  
  bool initialized_;
  //the bool means killed ! 
  std::map<uint32_t,std::pair<float,bool> > hbheRecHits_;
  std::map<uint32_t,std::pair<float,bool> > hoRecHits_;
  std::map<uint32_t,std::pair<float,bool> > hfRecHits_;

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
