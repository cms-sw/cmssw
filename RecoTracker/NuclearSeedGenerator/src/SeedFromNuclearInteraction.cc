#include "RecoTracker/NuclearSeedGenerator/interface/SeedFromNuclearInteraction.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SeedFromNuclearInteraction::SeedFromNuclearInteraction(const edm::EventSetup& es, const edm::ParameterSet& iConfig) : 
rescaleDirectionFactor(iConfig.getParameter<double>("rescaleDirectionFactor")),
rescalePositionFactor(iConfig.getParameter<double>("rescalePositionFactor")) {

  edm::ESHandle<Propagator>  thePropagatorHandle;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",thePropagatorHandle);
  thePropagator = &(*thePropagatorHandle);
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  es.get<TransientRecHitRecord>().get("WithTrackAngle",theBuilder);
  isValid_=true;
}

//----------------------------------------------------------------------
void SeedFromNuclearInteraction::setMeasurements(const TM& tmAtInteractionPoint, const TM& outerTM) {

       // increase errors on the initial TSOS
       updatedTSOS = stateWithError(tmAtInteractionPoint.updatedState());
       theNewTM = &outerTM;

       // delete pointer to TrackingRecHits
       _hits.clear();

       // get the inner and outer transient TrackingRecHits
       innerHit = tmAtInteractionPoint.recHit().get();
       outerHit = outerTM.recHit().get();

       _hits.push_back( innerHit->hit()->clone() );
       _hits.push_back( outerHit->hit()->clone() );

       isValid_ = construct();
}

//----------------------------------------------------------------------
TrajectoryStateOnSurface SeedFromNuclearInteraction::stateWithError(const TSOS& state) const {
   // Modification of the momentum = ~infinite
   LocalTrajectoryParameters ltp = state.localParameters();
   AlgebraicVector5 v = ltp.vector();
   v[0] = 1E-8;
   LocalTrajectoryParameters newltp(v, ltp.pzSign(), true);

   AlgebraicSymMatrix55 m(state.localError().matrix());
   m(0,0)=m(0,0)*1E6; 
   m(1,1)=m(1,1)*rescaleDirectionFactor*rescaleDirectionFactor;
   m(2,2)=m(2,2)*rescalePositionFactor*rescalePositionFactor;
   return TSOS(newltp, m, state.surface(), &(state.globalParameters().magneticField()), state.surfaceSide());
}

//----------------------------------------------------------------------
bool SeedFromNuclearInteraction::construct() {

   TSOS outerState = theNewTM->updatedState();
   updatedTSOS = thePropagator->propagate(updatedTSOS, outerState.surface());

   if ( !updatedTSOS.isValid()) { 
           LogDebug("SeedFromNuclearInteraction") << "Propagated state is invalid" << "\n";
           return false; }

   KFUpdator     theUpdator;
   updatedTSOS = theUpdator.update( updatedTSOS, *(theNewTM->recHit()));

   if ( !updatedTSOS.isValid()) { 
          LogDebug("SeedFromNuclearInteraction") << "Propagated state is invalid" << "\n";
          return false; }

   TrajectoryStateTransform transformer;

   pTraj = boost::shared_ptr<PTrajectoryStateOnDet>(transformer.persistentState(updatedTSOS, outerHit->geographicalId().rawId()));
   return true;
}
