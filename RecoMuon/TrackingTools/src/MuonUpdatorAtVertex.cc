/**  \class MuonUpdatorAtVertex
 *
 *   Extrapolate a muon trajectory to 
 *   a given vertex and 
 *   apply a vertex constraint
 *
 *   $Date: 2006/08/30 20:36:07 $
 *   $Revision: 1.13 $
 *
 *   \author   N. Neumeister         Purdue University
 *   \author   C. Liu                Purdue University 
 *
 */

#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoMuon/TrackingTools/interface/VertexRecHit.h"
#include "RecoMuon/TrackingTools/interface/DummyDet.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"

using namespace edm;
using namespace std;

//----------------
// Constructors --
//----------------

MuonUpdatorAtVertex::MuonUpdatorAtVertex(const edm::ParameterSet& par, const MuonServiceProxy* service) : 
  theService(service),
  theExtrapolator( new TransverseImpactPointExtrapolator() ),
  theUpdator(new KFUpdator()),
  theEstimator(new Chi2MeasurementEstimator(150.)) {

  theOutPropagatorName = par.getParameter<string>("OutPropagator");
  theInPropagatorName = par.getParameter<string>("InPropagator");

  // assume beam spot position with nominal errors
  // sigma(x) = sigma(y) = 15 microns
  // sigma(z) = 5.3 cm
  theVertexPos = GlobalPoint(0.0,0.0,0.0);
  theVertexErr = GlobalError(0.00000225, 0., 0.00000225, 0., 0., 28.09);
  
}


//
// default constructor, set propagator name as SteppingHelixPropagator
//
MuonUpdatorAtVertex::MuonUpdatorAtVertex() :
         theExtrapolator( new TransverseImpactPointExtrapolator() ),
         theUpdator(new KFUpdator()),
         theEstimator(new Chi2MeasurementEstimator(150.)) {

  theOutPropagatorName = "SteppingHelixPropagatorAny";
  theInPropagatorName = "PropagatorWithMaterial";

  // assume beam spot position with nominal errors
  // sigma(x) = sigma(y) = 15 microns
  // sigma(z) = 5.3 cm
  theVertexPos = GlobalPoint(0.0,0.0,0.0);
  theVertexErr = GlobalError(0.00000225, 0., 0.00000225, 0., 0., 28.09);

}


//
//
//
// MuonUpdatorAtVertex::MuonUpdatorAtVertex(const Propagator& prop) :
//          theExtrapolator(new TransverseImpactPointExtrapolator(prop)),
//          theUpdator(new KFUpdator()),
//          theEstimator(new Chi2MeasurementEstimator(150.)) {

//   thePropagator = prop.clone();
//   // assume beam spot position with nominal errors
//   // sigma(x) = sigma(y) = 15 microns
//   // sigma(z) = 5.3 cm
//   theVertexPos = GlobalPoint(0.0,0.0,0.0);
//   theVertexErr = GlobalError(0.00000225, 0., 0.00000225, 0., 0., 28.09);

// }

//---------------
// Destructor  --
//---------------
MuonUpdatorAtVertex::~MuonUpdatorAtVertex() {
   
  if (theEstimator) delete theEstimator;
  if (theUpdator) delete theUpdator;
  if (theExtrapolator) delete theExtrapolator;


}

// get Propagator for outside tracker, SteppingHelixPropagator as default
// anyDirection
auto_ptr<Propagator> MuonUpdatorAtVertex::propagator() const{

    auto_ptr<Propagator> smartPropagator(new SmartPropagator(*theService->propagator(theInPropagatorName),
							     *theService->propagator(theOutPropagatorName),
							     &*theService->magneticField() ));
    return smartPropagator;
}

//
//
//
void MuonUpdatorAtVertex::setVertex(const GlobalPoint& p, const GlobalError& e) {

  theVertexPos = p;
  theVertexErr = e;

}


//
//
//
MuonVertexMeasurement MuonUpdatorAtVertex::update(const TrajectoryStateOnSurface& tsos) const {
  
  if ( !tsos.isValid() ) {
    edm::LogError("MuonUpdatorAtVertex") << "Error invalid TrajectoryStateOnSurface";
    return MuonVertexMeasurement();
  }
  
  // get state at outer tracker surface
  TrajectoryStateOnSurface trackerState = stateAtTracker(tsos);

  // inside the tracker we can use Gtf propagator
  TrajectoryStateOnSurface ipState = theExtrapolator->extrapolate(trackerState,theVertexPos, *propagator() );
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
    //    GenericTransientTrackingRecHit* recHit = GenericTransientTrackingRecHit(&(det.geomDet()), trecHit);
    TransientTrackingRecHit::RecHitPointer recHit = GenericTransientTrackingRecHit::build((&(det.geomDet())), trecHit);


    std::pair<bool,double> pairChi2 = theEstimator->estimate(ipState, *recHit);

    chi2 = pairChi2.second;

    vertexState = theUpdator->update(ipState, *recHit);

//    det.addRecHit(recHit);
// measurements methods no longer exits for det
    vertexMeasurement = TrajectoryMeasurement(ipState,vertexState,&*recHit,chi2);

  }
  return MuonVertexMeasurement(trackerState,ipState,vertexState,vertexMeasurement,chi2);

}


//
//
//
TrajectoryStateOnSurface MuonUpdatorAtVertex::stateAtTracker(const TrajectoryStateOnSurface& tsos) const {

  if ( !tsos.isValid() ) {
    edm::LogError("MuonUpdatorAtVertex") << "Error invalid TrajectoryStateOnSurface";
    return TrajectoryStateOnSurface();
  }
  
  // get state at outer tracker surface
  StateOnTrackerBound tracker( &*propagator() );

  TrajectoryStateOnSurface result = tracker(tsos);

  return result;

}
