#include "RecoTracker/NuclearSeedGenerator/interface/SeedFromNuclearInteraction.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TangentCircle.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SeedFromNuclearInteraction::SeedFromNuclearInteraction(const edm::EventSetup& es, const edm::ParameterSet& iConfig) : 
rescaleDirectionFactor(iConfig.getParameter<double>("rescaleDirectionFactor")),
rescalePositionFactor(iConfig.getParameter<double>("rescalePositionFactor")),
rescaleCurvatureFactor(iConfig.getParameter<double>("rescaleCurvatureFactor")) {

  edm::ESHandle<Propagator>  thePropagatorHandle;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",thePropagatorHandle);
  thePropagator = &(*thePropagatorHandle);
  es.get<TransientRecHitRecord>().get("WithTrackAngle",theBuilder);
  isValid_=true;

  es.get<TrackerDigiGeometryRecord> ().get (pDD);
}

//----------------------------------------------------------------------
void SeedFromNuclearInteraction::setMeasurements(const TM& tmAtInteractionPoint, const TM& theNewTM) {

       // delete pointer to TrackingRecHits
       _hits.clear();

       // get the inner and outer transient TrackingRecHits
       innerHit = tmAtInteractionPoint.recHit().get();
       outerHit = theNewTM.recHit().get();

       // _hits.push_back( innerHit->hit()->clone() ); // put in comment to avoid taking into account first hit
                                                       // in CTFit
       _hits.push_back( outerHit->hit()->clone() );

       // increase errors on the initial TSOS
       updatedTSOS = stateWithError(tmAtInteractionPoint.updatedState());
       outerTM = &theNewTM;


       isValid_ = construct();
}

//----------------------------------------------------------------------
TrajectoryStateOnSurface SeedFromNuclearInteraction::stateWithError(const TSOS& state) const {
   // Calculation of the curvature assuming that the secondary track has the same direction
   // than the primary track and pass through the inner and outer hits.
   GlobalVector direction = state.globalDirection();
   GlobalPoint inner = state.globalPosition();
   GlobalPoint outer = pDD->idToDet(outerHit->geographicalId())->surface().toGlobal(outerHit->localPosition());
   TangentCircle circle(direction, inner, outer);
   double transverseCurvature = 1/circle.rho();

   // Get the global parameters of the trajectory
   GlobalTrajectoryParameters gtp(state.globalPosition(), direction, transverseCurvature, 0, &(state.globalParameters().magneticField()));

   // Rescale the error matrix
   TSOS result(gtp, state.cartesianError(), state.surface(), state.surfaceSide());
   AlgebraicSymMatrix55 m(result.localError().matrix());
   m(0,0)=m(0,0)*rescaleCurvatureFactor*rescaleCurvatureFactor;
   m(1,1)=m(1,1)*rescaleDirectionFactor*rescaleDirectionFactor;
   m(2,2)=m(2,2)*rescaleDirectionFactor*rescaleDirectionFactor;
   m(3,3)=m(3,3)*rescalePositionFactor*rescalePositionFactor;
   m(4,4)=m(4,4)*rescalePositionFactor*rescalePositionFactor;

   return TSOS(result.localParameters(), LocalTrajectoryError(m), result.surface(), &(result.globalParameters().magneticField()), result.surfaceSide());
}

//----------------------------------------------------------------------
bool SeedFromNuclearInteraction::construct() {

   TSOS outerState = outerTM->updatedState();
   updatedTSOS = thePropagator->propagate(updatedTSOS, outerState.surface());

   if ( !updatedTSOS.isValid()) { 
           LogDebug("SeedFromNuclearInteraction") << "Propagated state is invalid" << "\n";
           return false; }

   KFUpdator     theUpdator;
   updatedTSOS = theUpdator.update( updatedTSOS, *(outerTM->recHit()));

   if ( !updatedTSOS.isValid()) { 
          LogDebug("SeedFromNuclearInteraction") << "Propagated state is invalid" << "\n";
          return false; }

   TrajectoryStateTransform transformer;

   pTraj = boost::shared_ptr<PTrajectoryStateOnDet>(transformer.persistentState(updatedTSOS, outerHit->geographicalId().rawId()));
   return true;
}
