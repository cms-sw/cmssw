#include "RecoPixelVertexing/PixelLowPtUtilities/interface/PixelTrackSeedGenerator.h"

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

#include <algorithm>

/*****************************************************************************/
PixelTrackSeedGenerator::PixelTrackSeedGenerator
  (const edm::EventSetup& es)
{
   // Get the magnetic field
  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  theMagField = magField.product();

  // Get the propagator
  edm::ESHandle<Propagator> thePropagatorHandle;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",
                                       thePropagatorHandle);
  thePropagator = &(*thePropagatorHandle);

  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();

  // Get transient rechit builder
  edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
  es.get<TransientRecHitRecord>().get("WithoutRefit",ttrhbESH);
  theTTRecHitBuilder = ttrhbESH.product();
}

/*****************************************************************************/
PixelTrackSeedGenerator::~PixelTrackSeedGenerator()
{
}

/*****************************************************************************/
double PixelTrackSeedGenerator::getRadius(const TrackingRecHit& recHit)
{
  DetId id = recHit.geographicalId();
  LocalPoint  lpos = recHit.localPosition();
  GlobalPoint gpos = theTracker->idToDet(id)->toGlobal(lpos);
  return gpos.perp2();
}

/*****************************************************************************/
void PixelTrackSeedGenerator::sortRecHits
  (vector<pair<double,int> >& recHits)
{
  bool change;

  do
  {
    change = false;

    for(unsigned int i = 0; i < recHits.size() - 1 ; i++)
    if(recHits[i].first > recHits[i+1].first)
    {
      pair<double,int> r = recHits[i];
      recHits[i] = recHits[i+1];
      recHits[i+1] = r;

      change = true;
    } 
  }
  while(change);
}

/*****************************************************************************/
TrajectorySeed PixelTrackSeedGenerator::seed(const reco::Track& track)
{
  // Get free trajectory state
  GlobalPoint vertex(track.vertex().x(),
                     track.vertex().y(),
                     track.vertex().z());
  GlobalVector momentum(track.momentum().x(),
                        track.momentum().y(),
                        track.momentum().z());

//  cerr << " SEED " << track.vertex() << " " << track.momentum() << endl;

  GlobalTrajectoryParameters gtp(vertex,momentum,track.charge(), theMagField);
  AlgebraicSymMatrix C(5,1); CurvilinearTrajectoryError cte(C);
  FreeTrajectoryState fts(gtp,cte);

  // Sort rechits, radius increasing
  vector<pair<double,int> > radius; int i = 0;
  for(trackingRecHit_iterator recHit = track.recHitsBegin();
                              recHit!= track.recHitsEnd(); recHit++)
    if((*recHit)->isValid())
      radius.push_back(pair<double,int>(getRadius(**recHit),i++));

  sortRecHits(radius); 

  // Initialize
  KFUpdator theUpdator;

  TrajectoryStateOnSurface lastState;
  DetId lastId;
  edm::OwnVector<TrackingRecHit> hits;

  for(vector<pair<double,int> >::iterator ir = radius.begin();
                                          ir!= radius.end(); ir++)
  {
    // Get detector
    TrackingRecHitRef recHit = track.recHit(ir->second);
    DetId id = recHit->geographicalId();
    const PixelGeomDetUnit* pixelDet =
      dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));

    TrajectoryStateOnSurface state;
    if(ir == radius.begin())
      state = thePropagator->propagate(fts, pixelDet->surface());
    else
      state = thePropagator->propagate(lastState, pixelDet->surface());

//    cerr << " state = " << state.globalPosition()
//                 << " " << state.globalDirection() << endl;

    if(! state.isValid())
    {
      break;
    }

//    cerr << " Add hit " << pixelDet->toGlobal(recHit->localPosition()) << endl;

    TransientTrackingRecHit::RecHitPointer transientRecHit =
      theTTRecHitBuilder->build(&(*recHit));

    if(transientRecHit->isValid())
    {
      hits.push_back(recHit->clone());
      lastState = theUpdator.update(state, *transientRecHit);
      lastId    = id;
//     cerr << " lstte = " << lastState.globalPosition()
//                 << " " << lastState.globalDirection() << endl;

    }
  }

  TrajectorySeed trajSeed;

  if(lastState.isValid())
  {
    TrajectoryStateTransform transformer;
    PTrajectoryStateOnDet *PTraj=
              transformer.persistentState(lastState, lastId.rawId());

    trajSeed = TrajectorySeed(*PTraj,hits,alongMomentum);
  }
/*
  else
    cerr << " last state not vaild" << endl;
*/

  return trajSeed;
}

