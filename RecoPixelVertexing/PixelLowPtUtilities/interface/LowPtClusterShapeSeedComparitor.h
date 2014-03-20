#ifndef _LowPtClusterShapeSeedComparitor_h_
#define _LowPtClusterShapeSeedComparitor_h_


#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

class TrackerTopology;

namespace edm { class ParameterSet; class EventSetup; }

class LowPtClusterShapeSeedComparitor : public SeedComparitor
{
 public:
  LowPtClusterShapeSeedComparitor(const edm::ParameterSet& ps){}
  virtual ~LowPtClusterShapeSeedComparitor(){}
  virtual void init(const edm::EventSetup& es) ;
  virtual bool compatible(const SeedingHitSet  &hits, const TrackingRegion & region) const ;
  virtual bool compatible(const TrajectorySeed &seed) const { return true; }
  virtual bool compatible(const TrajectoryStateOnSurface &,  
                          SeedingHitSet::ConstRecHitPointer hit) const { return true; }
  virtual bool compatible(const SeedingHitSet  &hits, 
                          const GlobalTrajectoryParameters &helixStateAtVertex,
                          const FastHelix                  &helix,
                          const TrackingRegion & region) const { return true; }
  virtual bool compatible(const SeedingHitSet  &hits, 
                          const GlobalTrajectoryParameters &straightLineStateAtVertex,
                          const TrackingRegion & region) const { return true; }

 private:
   /// something
   edm::ESHandle<ClusterShapeHitFilter> theShapeFilter;
   edm::ESHandle<TrackerTopology> theTTopo;
};

#endif

