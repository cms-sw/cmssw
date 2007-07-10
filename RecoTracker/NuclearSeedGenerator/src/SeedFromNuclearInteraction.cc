#include "RecoTracker/NuclearSeedGenerator/interface/SeedFromNuclearInteraction.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define RESCALE_FACTOR 10


SeedFromNuclearInteraction::SeedFromNuclearInteraction(const Propagator* prop, const TrackerGeometry* geom, const edm::ParameterSet& iConfig):
    rescaleDirectionFactor(iConfig.getParameter<double>("rescaleDirectionFactor")),
    rescalePositionFactor(iConfig.getParameter<double>("rescalePositionFactor")),
    rescaleCurvatureFactor(iConfig.getParameter<double>("rescaleCurvatureFactor")),
    ptMin(iConfig.getParameter<double>("ptMin")),
    thePropagator(prop), theTrackerGeom(geom)
    {
         isValid_=true;
    }

//----------------------------------------------------------------------
void SeedFromNuclearInteraction::setMeasurements(const TSOS& inner_TSOS, ConstRecHitPointer ihit, ConstRecHitPointer ohit) {

       // delete pointer to TrackingRecHits
       theHits.clear();

       // get the inner and outer transient TrackingRecHits
       innerHit_ = ihit;
       outerHit_ = ohit;

       //theHits.push_back(  inner_TM.recHit() ); // put temporarily - TODO: remove this line
       theHits.push_back(  outerHit_  );

       initialTSOS_.reset( new TrajectoryStateOnSurface(inner_TSOS) );

       // calculate the initial FreeTrajectoryState.
       freeTS_.reset(stateWithError());

       // check transverse momentum
       if(freeTS_->momentum().perp() < ptMin) { isValid_ = false; }
       else {
          // convert freeTS_ into a persistent TSOS on the outer surface
          isValid_ = construct(); }
}
//----------------------------------------------------------------------
void SeedFromNuclearInteraction::setMeasurements(TangentHelix& thePrimaryHelix, const TSOS& inner_TSOS, ConstRecHitPointer ihit, ConstRecHitPointer ohit) {

       // delete pointer to TrackingRecHits
       theHits.clear();

       // get the inner and outer transient TrackingRecHits
       innerHit_ = ihit;
       outerHit_ = ohit;

       GlobalPoint innerPos = theTrackerGeom->idToDet(innerHit_->geographicalId())->surface().toGlobal(innerHit_->localPosition());
       GlobalPoint outerPos = theTrackerGeom->idToDet(outerHit_->geographicalId())->surface().toGlobal(outerHit_->localPosition());

       TangentHelix helix(thePrimaryHelix, outerPos, innerPos);

       theHits.push_back( innerHit_ );
       theHits.push_back( outerHit_ );

       TSOS rescaled_TSOS = inner_TSOS;
       rescaled_TSOS.rescaleError(1.0/((double)(RESCALE_FACTOR)));
       initialTSOS_.reset( new TrajectoryStateOnSurface(rescaled_TSOS) );

       // calculate the initial FreeTrajectoryState from the inner and outer TM assuming that the helix equation is already known.
       freeTS_.reset(stateWithError(helix)); 

       if(freeTS_->momentum().perp() < ptMin) { isValid_ = false; }
       else {
          // convert freeTS_ into a persistent TSOS on the outer surface
          isValid_ = construct(); }
}
//----------------------------------------------------------------------
FreeTrajectoryState* SeedFromNuclearInteraction::stateWithError() const {

   // Calculation of the helix assuming that the secondary track has the same direction
   // than the primary track and pass through the inner and outer hits.
   GlobalVector direction = initialTSOS_->globalDirection();
   GlobalPoint inner = initialTSOS_->globalPosition();
   TangentHelix helix(direction, inner, outerHitPosition());

   return stateWithError(helix);
}
//----------------------------------------------------------------------
FreeTrajectoryState* SeedFromNuclearInteraction::stateWithError(TangentHelix& helix) const {

//   typedef TkRotation<float> Rotation;

   GlobalVector dirAtVtx = helix.directionAtVertex();
   const MagneticField& mag = initialTSOS_->globalParameters().magneticField();

   // Get the global parameters of the trajectory
   // we assume that the magnetic field at the vertex is equal to the magnetic field at the inner TM.
   GlobalTrajectoryParameters gtp(helix.vertexPoint(), dirAtVtx , helix.charge(mag.inTesla(helix.vertexPoint()).z())/helix.rho(), 0, &mag);

   // Error matrix in a frame where z is in the direction of the track at the vertex
   //AlgebraicSymMatrix55 m = ROOT::Math::SMatrixIdentity();
   AlgebraicSymMatrix55 m(initialTSOS_->curvilinearError().matrix());
   double curvatureError = helix.curvatureError();
   m(0,0)=curvatureError*curvatureError;
   m(1,1)=m(1,1)*rescaleDirectionFactor*rescaleDirectionFactor;
   m(2,2)=m(2,2)*rescaleDirectionFactor*rescaleDirectionFactor;
   m(3,3)=m(3,3)*rescalePositionFactor*rescalePositionFactor;
   m(4,4)=m(4,4)*rescalePositionFactor*rescalePositionFactor;

/*
   //rotation around the z-axis by  -phi
   Rotation tmpRotz ( cos(dirAtVtx.phi()), -sin(dirAtVtx.phi()), 0., 
                        sin(dirAtVtx.phi()), cos(dirAtVtx.phi()), 0.,
                         0.,              0.,              1. );

   //rotation around y-axis by -theta
   Rotation tmpRoty ( cos(dirAtVtx.theta()), 0.,sin(dirAtVtx.theta()),
                               0.,              1.,              0.,
                              -sin(dirAtVtx.theta()), 0., cos(dirAtVtx.theta()) );

   Rotation position(m(0,0), 0, 0, 0, m(1,1), 0, 0, 0, m(2,2) );
   Rotation momentum(m(3,3), 0, 0, 0, m(4,4), 0, 0, 0, m(5,5) ); 

   // position = position * tmpRoty * tmpRotz
   // momentum = momentum * tmpRoty * tmpRotz
   position *= tmpRoty;   momentum *= tmpRoty; 
   position *= tmpRotz;   momentum *= tmpRotz; 

   m(0,0) = position.xx();
   m(1,0) = position.yx();
   m(2,0) = position.zx();
   m(0,1) = position.xy();
   m(1,1) = position.yy();
   m(2,1) = position.zy();
   m(0,2) = position.xz();
   m(1,2) = position.yz();
   m(2,2) = position.zz();
   m(3,3) = momentum.xx();
   m(4,3) = momentum.yx();
   m(5,3) = momentum.zx();
   m(3,4) = momentum.xy();
   m(4,4) = momentum.yy();
   m(5,4) = momentum.zy();
   m(3,5) = momentum.xz();
   m(4,5) = momentum.yz();
   m(5,5) = momentum.zz();
 */  


   return new FreeTrajectoryState( gtp, CurvilinearTrajectoryError(m) );
}

//----------------------------------------------------------------------
bool SeedFromNuclearInteraction::construct() {

   // loop on all hits in theHits
   KFUpdator                 theUpdator;

   const TrackingRecHit* hit = 0;

   for ( unsigned int iHit = 0; iHit < theHits.size(); iHit++) {
     hit = theHits[iHit]->hit();
     TrajectoryStateOnSurface state = (iHit==0) ? 
        thePropagator->propagate( *freeTS_, theTrackerGeom->idToDet(hit->geographicalId())->surface())
       : thePropagator->propagate( *updatedTSOS_, theTrackerGeom->idToDet(hit->geographicalId())->surface());

     if (!state.isValid()) return false; 
 
     const TransientTrackingRecHit::ConstRecHitPointer& tth = theHits[iHit]; 
     updatedTSOS_.reset( new TrajectoryStateOnSurface(theUpdator.update(state, *tth)) );

   } 

   TrajectoryStateTransform transformer;

   updatedTSOS_->rescaleError(RESCALE_FACTOR);

   pTraj = boost::shared_ptr<PTrajectoryStateOnDet>( transformer.persistentState(*updatedTSOS_, outerHitDetId().rawId()) );
   return true;
}

//----------------------------------------------------------------------
edm::OwnVector<TrackingRecHit>  SeedFromNuclearInteraction::hits() const { 
    recHitContainer      _hits;
    for( ConstRecHitContainer::const_iterator it = theHits.begin(); it!=theHits.end(); it++ ){
           _hits.push_back( it->get()->hit()->clone() );
    }
    return _hits; 
}
