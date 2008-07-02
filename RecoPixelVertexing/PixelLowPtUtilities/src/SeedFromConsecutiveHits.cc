#include "RecoPixelVertexing/PixelLowPtUtilities/interface/SeedFromConsecutiveHits.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"

using namespace std;

/*****************************************************************************/
SeedFromConsecutiveHits::SeedFromConsecutiveHits(
    const vector<const TrackingRecHit *> & hits,
    const GlobalPoint& vertexPos,
    const GlobalError& vertexErr,
    const edm::EventSetup& es,
    const edm::ParameterSet& ps) 
  : isValid_(false)
{
  if(hits.size() < 2) return;

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  // build initial helix and FTS
  GlobalPoint inner =
    tracker->idToDet(hits[0]->geographicalId())->surface().toGlobal(hits[0]->localPosition());
  GlobalPoint outer =
    tracker->idToDet(hits[1]->geographicalId())->surface().toGlobal(hits[1]->localPosition());

  FastHelix helix(inner, outer, vertexPos, es);

  GlobalTrajectoryParameters kine = helix.stateAtVertex().parameters();
  float sinTheta = sin( kine.momentum().theta() );
  FreeTrajectoryState fts( kine, initialError( vertexPos, vertexErr, sinTheta));

  // get propagator
  edm::ESHandle<Propagator>  thePropagatorHandle;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",thePropagatorHandle);
  const Propagator*  thePropagator = &(*thePropagatorHandle);

  // get the transient builder
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = ps.getParameter<std::string>("TTRHBuilder");
  es.get<TransientRecHitRecord>().get(builderName,theBuilder);

  // get updator
  KFUpdator     theUpdator;

  TrajectoryStateOnSurface updatedState;
  const TrackingRecHit* hit = 0;
  for ( unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit];

GlobalPoint hitpos =
    tracker->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition());

    TrajectoryStateOnSurface state = (iHit==0) ? 
        thePropagator->propagate(fts,
          tracker->idToDet(hit->geographicalId())->surface())
      : thePropagator->propagate(updatedState,
          tracker->idToDet(hit->geographicalId())->surface());

    {
    LogTrace("MinBiasTracking")
        << "   [SeedFromConsecutiveHits] hit  #" << iHit+1
        << " " << HitInfo::getInfo(*hit);
    }

    if (!state.isValid())
    {
      LogTrace("MinBiasTracking")
        << "   [SeedFromConsecutiveHits] state #" << iHit+1
        << " out of "<< hits.size() << " not valid ";
      return;
    }

    outrhit = theBuilder.product()->build(hits[iHit]);

    updatedState = theUpdator.update(state, *outrhit);

    _hits.push_back(hit->clone());
      
  } 
  PTraj = boost::shared_ptr<PTrajectoryStateOnDet>(
    transformer.persistentState(updatedState, hit->geographicalId().rawId()) );

  isValid_ = true;
}

/*****************************************************************************/
CurvilinearTrajectoryError SeedFromConsecutiveHits::
   initialError( const GlobalPoint& vertexPos, const GlobalError& vertexErr, float sinTheta)  
{
  AlgebraicSymMatrix C(5,1);

  float zErr = vertexErr.czz(); 
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy 
  C[3][3] = transverseErr;
  C[4][4] = zErr*sinTheta;
  
  return CurvilinearTrajectoryError(C);
}

