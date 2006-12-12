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
  class signalHit;
  unsigned createVectorsOfCells(const edm::EventSetup &es);
  unsigned createVectorOfSubdetectorCells( const CaloGeometry&,int subdetn,std::vector<uint32_t>&);
  void noisifySubdet(std::map<signalHit,float>& theMap, const std::vector<uint32_t>& thecells, unsigned ncells); 
  void noisify();
  void noisifyAndFill(uint32_t id,float energy, std::map<signalHit,float>& myHits);
  void loadPSimHits(const edm::Event & iEvent);
  
  void clean();

 private:
  class signalHit
  {
  public:
    signalHit(uint32_t val ,bool killed=false):val_(val),killed_(killed)
      {;
      }
      ~signalHit() {;};
      inline uint32_t operator()() const {return val_;};	
      inline uint32_t id() const {return val_;}
      inline bool operator<(const signalHit& comp) const {return val_<comp.id();}
      inline bool operator==(const signalHit& comp) const {return val_==comp.id();}
      inline bool killed() const {return killed_;}
  private:
      uint32_t val_;
      bool killed_;
  };

 private:
  double threshold_;
  double noise_;
  double hcalHotFraction_; 

  //  edm::ESHandle<CaloTowerConstituentsMap> calotowerMap_;
  bool initialized_;
  std::map<signalHit,float> hbheRecHits_;
  std::map<signalHit,float> hoRecHits_;
  std::map<signalHit,float> hfRecHits_;

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
