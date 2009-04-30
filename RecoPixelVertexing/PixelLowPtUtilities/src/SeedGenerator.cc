#include "RecoPixelVertexing/PixelLowPtUtilities/interface/SeedGenerator.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
//#include "RecoPixelVertexing/PixelLowPtUtilities/interface/SeedFromConsecutiveHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromProtoTrack.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"

#include <algorithm>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

using namespace std;

/*****************************************************************************/
SeedGenerator::SeedGenerator(const edm::EventSetup& es)
{
}

/*****************************************************************************/
SeedGenerator::~SeedGenerator()
{
}

/*****************************************************************************/

class SortByRadius : public std::binary_function<const TransientTrackingRecHit::ConstRecHitPointer &,
                                                 const TransientTrackingRecHit::ConstRecHitPointer &, bool>
{
 public:
  SortByRadius(){}

  bool operator() (const TransientTrackingRecHit::ConstRecHitPointer & h1,
                   const TransientTrackingRecHit::ConstRecHitPointer & h2) const
  {
    GlobalPoint gp1 = h1->globalPosition();
    GlobalPoint gp2 = h2->globalPosition();
    return (gp1.perp2() < gp2.perp2());
  };

};

/*****************************************************************************/
TrajectorySeed SeedGenerator::seed
 (const reco::Track& track, const edm::EventSetup& es,
  const edm::ParameterSet& ps)
{
  
  // get the transient builder
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = ps.getParameter<std::string>("TTRHBuilder");
  es.get<TransientRecHitRecord>().get(builderName,theBuilder);

  TransientTrackingRecHit::ConstRecHitContainer thits;
  
  for (unsigned int iHit = 0, nHits = track.recHitsSize();
       iHit < nHits; ++iHit)
  {  
    TrackingRecHitRef refHit = track.recHit(iHit);
    if(refHit->isValid())
      thits.push_back( theBuilder.product()->build(&*refHit));
  } 
  
  SeedingHitSet hitset(thits);
  
  sort(thits.begin(), thits.end(), SortByRadius());

  GlobalPoint vtx(track.vertex().x(),
                  track.vertex().y(),
                  track.vertex().z()); 
  double originRBound = 0.2;
  double originZBound = 0.2;
  GlobalError vtxerr( originRBound*originRBound, 0, originRBound*originRBound,
                                              0, 0, originZBound*originZBound );

  SeedFromConsecutiveHits seedFromHits(hitset, vtx, vtxerr, es, 0.0);

  if(seedFromHits.isValid()) 
  {
    return seedFromHits.TrajSeed();
  }
  else
  {
    return TrajectorySeed();
  }
}

