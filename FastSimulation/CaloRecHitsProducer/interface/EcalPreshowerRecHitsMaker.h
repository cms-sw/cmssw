#ifndef FastSimulation__EcalPreshowerRecHitsMaker__h
#define FastSimulation__EcalPreshowerRecHitsMaker__h

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FastSimulation/Utilities/interface/GaussianTail.h"
#include "FWCore/Utilities/interface/InputTag.h"
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

class EcalPreshowerRecHitsMaker
{
 public:

  EcalPreshowerRecHitsMaker(edm::ParameterSet const & p, 
			    const RandomEngine* random);

  ~EcalPreshowerRecHitsMaker();

  void loadEcalPreshowerRecHits(edm::Event &iEvent, ESRecHitCollection& esRecHits);
  void init(const edm::EventSetup &es);
  


 private:
  
  void loadPCaloHits(const edm::Event & iEvent);
  
  void clean();

  unsigned createVectorsOfCells(const edm::EventSetup &es);
  void noisifySubdet(std::map<uint32_t, std::pair<float,bool> >& theMap, const std::vector<uint32_t>& thecells, unsigned ncells);
  void noisifySignal(std::map<uint32_t,std::pair<float,bool> >& theMap); 
  void noisify();
  void Fill(uint32_t id,float energy, std::map<uint32_t,std::pair<float,bool> >& myHits,bool signal=true);

 private:
  edm::InputTag inputCol_;
  double threshold_;
  double noise_;
  double preshowerHotFraction_;
  bool initialized_;
  unsigned ncells_;
  std::map<uint32_t,std::pair<float,bool> > ecalsRecHits_;
  std::vector<uint32_t> escells_;
  const RandomEngine* random_;
  const GaussianTail* myGaussianTailGenerator_;
};

#endif
