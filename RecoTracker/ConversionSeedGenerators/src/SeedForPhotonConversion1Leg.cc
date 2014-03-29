#include "RecoTracker/ConversionSeedGenerators/interface/SeedForPhotonConversion1Leg.h"

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
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"

//#define mydebug_seed
template <class T> T sqr( T t) {return t*t;}

const TrajectorySeed * SeedForPhotonConversion1Leg::trajectorySeed(
    TrajectorySeedCollection & seedCollection,
    const SeedingHitSet & hits,
    const GlobalPoint & vertex,
    const GlobalVector & vertexBounds,
    float ptmin,
    const edm::EventSetup& es,
    float cotTheta, std::stringstream& ss)
{
  pss = &ss;
  if ( hits.size() < 2) return 0;

  GlobalTrajectoryParameters kine = initialKinematic(hits, vertex, es, cotTheta);
  float sinTheta = sin(kine.momentum().theta());

  CurvilinearTrajectoryError error = initialError(vertexBounds, ptmin,  sinTheta);
  FreeTrajectoryState fts(kine, error);

  return buildSeed(seedCollection,hits,fts,es); 
}


GlobalTrajectoryParameters SeedForPhotonConversion1Leg::initialKinematic(
      const SeedingHitSet & hits, 
      const GlobalPoint & vertexPos, 
      const edm::EventSetup& es,
      const float cotTheta) const
{
  GlobalTrajectoryParameters kine;

  SeedingHitSet::ConstRecHitPointer tth1 = hits[0];
  SeedingHitSet::ConstRecHitPointer tth2 = hits[1];

   // FIXME optimize: move outside loop
    edm::ESHandle<MagneticField> bfield;
    es.get<IdealMagneticFieldRecord>().get(bfield);
    float nomField = bfield->nominalValue();

  FastHelix helix(tth2->globalPosition(), tth1->globalPosition(), vertexPos, nomField, &*bfield, vertexPos);
  kine = helix.stateAtVertex();

  //force the pz/pt equal to the measured one
  if(fabs(cotTheta)<cotTheta_Max)
    kine = GlobalTrajectoryParameters(kine.position(),
				      GlobalVector(kine.momentum().x(),kine.momentum().y(),kine.momentum().perp()*cotTheta),
				      kine.charge(),
				      & kine.magneticField()
				      );
  else
    kine = GlobalTrajectoryParameters(kine.position(),
				      GlobalVector(kine.momentum().x(),kine.momentum().y(),kine.momentum().perp()*cotTheta_Max),
				      kine.charge(),
				      & kine.magneticField()
				      );

#ifdef mydebug_seed
  uint32_t detid;
  (*pss) << "[SeedForPhotonConversion1Leg] initialKinematic tth1 " ;
  detid=tth1->geographicalId().rawId();
  po.print(*pss, detid );
  (*pss) << " \t " << detid << " " << tth1->localPosition()  << " " << tth1->globalPosition()    ;
  detid= tth2->geographicalId().rawId();
  (*pss) << " \n\t tth2 ";
  po.print(*pss, detid );
  (*pss) << " \t " << detid << " " << tth2->localPosition()  << " " << tth2->globalPosition()  
	 << "\nhelix momentum " << kine.momentum() << " pt " << kine.momentum().perp() << " radius " << 1/kine.transverseCurvature(); 
#endif

  bool isBOFF =(0==nomField);;
  if (isBOFF && (theBOFFMomentum > 0)) {
    kine = GlobalTrajectoryParameters(kine.position(),
                              kine.momentum().unit() * theBOFFMomentum,
                              kine.charge(),
                              &*bfield);
  }
  return kine;
}



CurvilinearTrajectoryError SeedForPhotonConversion1Leg::
initialError( 
	     const GlobalVector& vertexBounds, 
	     float ptMin,  
	     float sinTheta) const
{
  // Set initial uncertainty on track parameters, using only P.V. constraint and no hit
  // information.
  GlobalError vertexErr( sqr(vertexBounds.x()), 0, 
			 sqr(vertexBounds.y()), 0, 0,
			 sqr(vertexBounds.z())
			 );
  
 
  AlgebraicSymMatrix55 C = ROOT::Math::SMatrixIdentity();

// FIXME: minC00. Prevent apriori uncertainty in 1/P from being too small, 
// to avoid instabilities.
// N.B. This parameter needs optimising ...
  float sin2th = sqr(sinTheta);
  float minC00 = 1.0;
  C[0][0] = std::max(sin2th/sqr(ptMin), minC00);
  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy
  C[3][3] = transverseErr;
  C[4][4] = zErr*sin2th + transverseErr*(1-sin2th);

  return CurvilinearTrajectoryError(C);
}

const TrajectorySeed * SeedForPhotonConversion1Leg::buildSeed(
    TrajectorySeedCollection & seedCollection,
    const SeedingHitSet & hits,
    const FreeTrajectoryState & fts,
    const edm::EventSetup& es) const
{
  // get tracker
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
  // get propagator
  edm::ESHandle<Propagator>  propagatorHandle;
  es.get<TrackingComponentsRecord>().get(thePropagatorLabel, propagatorHandle);
  const Propagator*  propagator = &(*propagatorHandle);
  
   // get cloner (FIXME: add to config)
  try { 
    auto TTRHBuilder = "WithTrackAngle";
    edm::ESHandle<TransientTrackingRecHitBuilder> builderH;
    es.get<TransientRecHitRecord>().get(TTRHBuilder, builderH);
    auto builder = (TkTransientTrackingRecHitBuilder const *)(builderH.product());
    cloner = (*builder).cloner();
  } catch(...) {
    auto TTRHBuilder = "hltESPTTRHBWithTrackAngle";
    edm::ESHandle<TransientTrackingRecHitBuilder> builderH;
    es.get<TransientRecHitRecord>().get(TTRHBuilder, builderH);
    auto builder = (TkTransientTrackingRecHitBuilder const *)(builderH.product());
    cloner = (*builder).cloner();
  }

  // get updator
  KFUpdator  updator;
  
  // Now update initial state track using information from seed hits.
  
  TrajectoryStateOnSurface updatedState;
  edm::OwnVector<TrackingRecHit> seedHits;
  
  const TrackingRecHit* hit = 0;
  for ( unsigned int iHit = 0; iHit < hits.size() && iHit<1; iHit++) {
    hit = hits[iHit];
    TrajectoryStateOnSurface state = (iHit==0) ? 
      propagator->propagate(fts,tracker->idToDet(hit->geographicalId())->surface())
      : propagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid()) return 0;
    
    SeedingHitSet::ConstRecHitPointer tth = hits[iHit]; 
    
    std::unique_ptr<BaseTrackerRecHit> newtth(refitHit( tth, state));

    
    if (!checkHit(state,&*newtth,es)) return 0;

    updatedState =  updator.update(state, *newtth);
    if (!updatedState.isValid()) return 0;
    
    seedHits.push_back(newtth.release());
#ifdef mydebug_seed
    uint32_t detid = hit->geographicalId().rawId();
    (*pss) << "\n[SeedForPhotonConversion1Leg] hit " << iHit;
    po.print(*pss, detid);
    (*pss) << " "  << detid << "\t lp " << hit->localPosition()
	   << " tth " << tth->localPosition() << " newtth " << newtth->localPosition() << " state " << state.globalMomentum().perp();
#endif
  } 
  
  
  PTrajectoryStateOnDet const & PTraj =
      trajectoryStateTransform::persistentState(updatedState, hit->geographicalId().rawId());
  
  seedCollection.push_back( TrajectorySeed(PTraj,seedHits,alongMomentum));
  return &seedCollection.back();
}

SeedingHitSet::RecHitPointer SeedForPhotonConversion1Leg::refitHit(
      SeedingHitSet::ConstRecHitPointer hit, 
      const TrajectoryStateOnSurface &state) const
{
  //const TransientTrackingRecHit* a= hit.get();
  //return const_cast<TransientTrackingRecHit*> (a);
  //This was modified otherwise the rechit will have just the local x component and local y=0
  // To understand how to modify for pixels

  //const TSiStripRecHit2DLocalPos* b = dynamic_cast<const TSiStripRecHit2DLocalPos*>(a);
  //return const_cast<TSiStripRecHit2DLocalPos*>(b);
  return (SeedingHitSet::RecHitPointer)(cloner(*hit,state));
}
