#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
/**  \class MuonUpdatorAtVertex
 *
 *   Extrapolate a muon trajectory to 
 *   a given vertex and 
 *   apply a vertex constraint
 *
 *   $Date: 2006/06/15 15:17:12 $
 *   $Revision: 1.2 $
 *
 *   \author   N. Neumeister         Purdue University
 *   \porthing author C. Liu         Purdue University 
 *
 */


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoMuon/TrackingTools/interface/VertexRecHit.h"
#include "RecoMuon/TrackingTools/interface/DummyDet.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Surface/interface/TkRotation.h"
#include "TrackingTools/PatternTools/interface/MediumProperties.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/Plane.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//----------------
// Constructors --
//----------------

MuonUpdatorAtVertex::MuonUpdatorAtVertex(const MagneticField* field) : 
         thePropagator(new SteppingHelixPropagator(field, oppositeToMomentum)), 
         theExtrapolator(new TransverseImpactPointExtrapolator()),
         theUpdator(new KFUpdator()),
         theEstimator(new Chi2MeasurementEstimator(150.)) {

  // assume beam spot position with nominal errors
  // sigma(x) = sigma(y) = 15 microns
  // sigma(z) = 5.3 cm
  theVertexPos = GlobalPoint(0.0,0.0,0.0);
  theVertexErr = GlobalError(0.00000225, 0., 0.00000225, 0., 0., 28.09);
  
}


MuonUpdatorAtVertex::MuonUpdatorAtVertex(const GlobalPoint p, const GlobalError e, const MagneticField* field) :
         theVertexPos(p),
         theVertexErr(e),
         thePropagator(new SteppingHelixPropagator(field, oppositeToMomentum)), 
         theExtrapolator(new TransverseImpactPointExtrapolator()),
         theUpdator(new KFUpdator()),
         theEstimator(new Chi2MeasurementEstimator(150.))
{ }


//---------------
// Destructor  --
//---------------
MuonUpdatorAtVertex::~MuonUpdatorAtVertex() {
   
  delete theEstimator;
  delete theUpdator;
  delete theExtrapolator;
  delete thePropagator;

}


//
//
//
MuonVertexMeasurement MuonUpdatorAtVertex::update(const TrajectoryStateOnSurface& tsos) const {
  
  if ( !tsos.isValid() ) {
    edm::LogInfo("MuonUpdatorAtVertex") << "Error invalid TrajectoryStateOnSurface";
    return MuonVertexMeasurement();
  }
  
  // propagate to the outer tracker surface (r = 123.3cm, halfLength = 293.5cm
  Cylinder surface = TrackerBounds::barrelBound(); 
  FreeTrajectoryState* ftsOftsos =tsos.freeState();
  std::pair<TrajectoryStateOnSurface, double> tsosAtBarrelTrackerPair =
  thePropagator->propagateWithPath(*ftsOftsos,surface);
    
  Plane negDisk = TrackerBounds::negativeEndcapDisk();
  std::pair<TrajectoryStateOnSurface, double> tsosAtNegTrackerPair =
  thePropagator->propagateWithPath(*ftsOftsos,negDisk);

  Plane posDisk = TrackerBounds::positiveEndcapDisk();
  std::pair<TrajectoryStateOnSurface, double> tsosAtPosTrackerPair =
  thePropagator->propagateWithPath(*ftsOftsos,posDisk);


  if ( tsosAtBarrelTrackerPair.second == 0. && 
       tsosAtBarrelTrackerPair.second == 0. &&
       tsosAtBarrelTrackerPair.second == 0. ) {
    edm::LogInfo("MuonUpdatorAtVertex")<<"Extrapolation to Tracker failed";
    return MuonVertexMeasurement();
  }
  TrajectoryStateOnSurface tsosAtTracker;
  if ( tsosAtNegTrackerPair.second != 0.)
     tsosAtTracker = tsosAtNegTrackerPair.first;
  if ( tsosAtPosTrackerPair.second != 0.)
     tsosAtTracker = tsosAtPosTrackerPair.first;
  if ( tsosAtBarrelTrackerPair.second != 0.)
     tsosAtTracker = tsosAtBarrelTrackerPair.first;
    
  // get state at outer tracker surface
  StateOnTrackerBound tracker(thePropagator);
  TrajectoryStateOnSurface trackerState = tracker(*tsosAtTracker.freeState());
  
  // inside the tracker we can use Gtf propagator
  TrajectoryStateOnSurface ipState = theExtrapolator->extrapolate(tsosAtTracker,theVertexPos);

  TrajectoryStateOnSurface vertexState;
  TrajectoryMeasurement vertexMeasurement;
  double chi2 = 0.0;
  
  if ( ipState.isValid() ) {

    // convert global error to 2D error matrix in the local frame of the tsos surface
    const Surface& surf = ipState.surface();

    ErrorFrameTransformer tran;
    LocalError err2D = tran.transform(theVertexErr,surf);
    // now construct a surface centred on the vertex and 
    // perpendicular to the trajectory
    // try to make BoundPlane identical to tsos surface
    const BoundPlane* plane = dynamic_cast<const BoundPlane*>(&surf);
    if ( plane == 0 ) {
      plane = new BoundPlane(surf.position(),surf.rotation());
    }

    DummyDet det(plane);

    const VertexRecHit* vrecHit = new VertexRecHit(LocalPoint(0.,0.),err2D); //FIXME
    const TrackingRecHit* trecHit = (*vrecHit).hit();
    GenericTransientTrackingRecHit* recHit = new GenericTransientTrackingRecHit(&(det.geomDet()), trecHit);

    std::pair<bool,double> pairChi2 = theEstimator->estimate(ipState, *recHit);

    chi2=pairChi2.second;

    vertexState = theUpdator->update(ipState, *recHit);

//    det.addRecHit(recHit);
// measurements methods no longer exits for det
    vertexMeasurement = TrajectoryMeasurement(ipState,vertexState,recHit,chi2);

  }


  return MuonVertexMeasurement(trackerState,ipState,vertexState,vertexMeasurement,chi2);

}

