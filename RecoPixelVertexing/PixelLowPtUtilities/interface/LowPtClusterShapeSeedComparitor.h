#ifndef _LowPtClusterShapeSeedComparitor_h_
#define _LowPtClusterShapeSeedComparitor_h_


#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

class TrackerTopology;

namespace edm { class ParameterSet; class EventSetup; }

class LowPtClusterShapeSeedComparitor : public SeedComparitor
{
 public:
  LowPtClusterShapeSeedComparitor(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
  virtual ~LowPtClusterShapeSeedComparitor(){}
  virtual void init(const edm::Event& e, const edm::EventSetup& es) ;
  virtual bool compatible(const SeedingHitSet  &hits) const ;
  virtual bool compatible(const TrajectoryStateOnSurface &,  
                          SeedingHitSet::ConstRecHitPointer hit) const { return true; }
  virtual bool compatible(const SeedingHitSet  &hits, 
                          const GlobalTrajectoryParameters &helixStateAtVertex,
                          const FastHelix                  &helix) const { return true; }

 private:
   /// something
   edm::ESHandle<ClusterShapeHitFilter> theShapeFilter;
   edm::ESHandle<TrackerTopology> theTTopo;
   edm::EDGetTokenT<SiPixelClusterShapeCache> thePixelClusterShapeCacheToken;
   edm::Handle<SiPixelClusterShapeCache> thePixelClusterShapeCache;
   std::string theShapeFilterLabel_;
};

#endif

