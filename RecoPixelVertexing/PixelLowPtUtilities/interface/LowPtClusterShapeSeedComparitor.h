#ifndef _LowPtClusterShapeSeedComparitor_h_
#define _LowPtClusterShapeSeedComparitor_h_


#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"


namespace edm { class ParameterSet; class EventSetup; }

class LowPtClusterShapeSeedComparitor : public SeedComparitor
{
 public:
  LowPtClusterShapeSeedComparitor(const edm::ParameterSet& ps){}
  virtual ~LowPtClusterShapeSeedComparitor(){}
  virtual bool compatible(const SeedingHitSet &hits, const edm::EventSetup &es);

 private:
};

#endif

