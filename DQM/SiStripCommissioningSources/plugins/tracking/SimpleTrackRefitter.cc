#include "DQM/SiStripCommissioningSources/plugins/tracking/SimpleTrackRefitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h>
#include <TrackingTools/PatternTools/interface/Trajectory.h>
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include <vector>

typedef TrajectoryStateOnSurface     TSOS;

SimpleTrackRefitter::SimpleTrackRefitter(const edm::ParameterSet& iConfig):
thePropagator(0),thePropagatorOp(0),theUpdator(0),theEstimator(0),RHBuilder(0),theSmoother(0),theFitter(0),tsTransform(),conf_(iConfig)
{
  isCosmics_ = conf_.getParameter<bool>("cosmic");
}

SimpleTrackRefitter::~SimpleTrackRefitter()
{

}

void SimpleTrackRefitter::initServices(const bool& seedAlongMomentum)
{
  // dynamic services for cosmic refit
  if(seedAlongMomentum) {
    thePropagator   = new PropagatorWithMaterial(alongMomentum,0.1057,magfield );
    thePropagatorOp = new PropagatorWithMaterial(oppositeToMomentum,0.1057,magfield );
  }
  else {
    thePropagator   = new PropagatorWithMaterial(oppositeToMomentum,0.1057,magfield );
    thePropagatorOp = new PropagatorWithMaterial(alongMomentum,0.1057,magfield );
  }
  theUpdator=         new KFUpdator();
  theEstimator=       new Chi2MeasurementEstimator(30);
  theFitter=          new KFTrajectoryFitter(*thePropagator,
					     *theUpdator,	
					     *theEstimator) ;
  theSmoother=        new KFTrajectorySmoother(*thePropagatorOp,
 					       *theUpdator,	
 					       *theEstimator);
}

void SimpleTrackRefitter::setServices(const edm::EventSetup& es)
{
  // get geometry
  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker);
  // get magnetic field
  edm::ESHandle<MagneticField> esmagfield;
  es.get<IdealMagneticFieldRecord>().get(esmagfield);
  magfield=&(*esmagfield);
  // get the builder
  edm::ESHandle<TransientTrackingRecHitBuilder> builder;
  LogDebug("SimpleTrackRefitter") << "get also the TransientTrackingRecHitBuilder" << "\n";
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  es.get<TransientRecHitRecord>().get(builderName,builder);
  RHBuilder=&(*builder);
  if(isCosmics_) return;
  // get the fitter
  edm::ESHandle<TrajectoryFitter> fitter;
  LogDebug("SimpleTrackRefitter") << "get the fitter from the ES" << "\n";
  std::string fitterName = conf_.getParameter<std::string>("Fitter");   
  es.get<TrajectoryFitter::Record>().get(fitterName,fitter);
  theFitter=&(*fitter);
  // get the propagator
  edm::ESHandle<Propagator> propagator;
  LogDebug("SimpleTrackRefitter") << "get also the propagator" << "\n";
  std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
  es.get<TrackingComponentsRecord>().get(propagatorName,propagator);
  thePropagator=&(*propagator);
}

void SimpleTrackRefitter::destroyServices()
{
  if(thePropagator) { delete thePropagator; thePropagator=0; }
  if(thePropagatorOp) { delete thePropagatorOp; thePropagatorOp =0; }
  if(theUpdator) { delete theUpdator; theUpdator=0; }
  if(theEstimator) { delete theEstimator; theEstimator=0; }
  if(theSmoother) { delete theSmoother; theSmoother=0; }
  if(theFitter) { delete theFitter; theFitter=0; }
}

std::vector<Trajectory> SimpleTrackRefitter::refitTrack(const reco::Track& theT, const uint32_t ExcludedDetid)
{
  // convert the TrackingRecHit vector to a TransientTrackingRecHit vector
  LogDebug("SimpleTrackRefitter") << "Start\n";
  TransientTrackingRecHit::RecHitContainer tmp;
  for (trackingRecHit_iterator i=theT.recHitsBegin();i!=theT.recHitsEnd(); i++) {
    if ( (*i)->geographicalId().det() == DetId::Tracker ) {
      if((*i)->isValid()) {
        if((*i)->geographicalId().rawId()!=ExcludedDetid) {
	  tmp.push_back(RHBuilder->build(&**i ));
	} else {
	  InvalidTrackingRecHit* irh = new InvalidTrackingRecHit((*i)->geographicalId(), TrackingRecHit::inactive);
	  tmp.push_back(RHBuilder->build(irh));
	  delete irh;
        }
      } else {
        tmp.push_back(RHBuilder->build(&**i ));
      }
    }
  }
  LogDebug("SimpleTrackRefitter") << "Transient rechit filled" << "\n";
  // sort rechits alongmomentum
  TransientTrackingRecHit::RecHitContainer hits;
  const TransientTrackingRecHit::ConstRecHitPointer *firstHit = NULL;
  for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.begin(); it!=tmp.end();it++){
    if ((**it).isValid()) {
      firstHit = &(*it);
      break;
    }
  }
  const TransientTrackingRecHit::ConstRecHitPointer *lastHit = NULL;
  for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.end()-1; it!=tmp.begin()-1;it--){
    if ((**it).isValid()) {
      lastHit= &(*it);
      break;
    }
  }
  if ((*firstHit)->globalPosition().mag2() > ((*lastHit)->globalPosition().mag2()) ){
    for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.end()-1;it!=tmp.begin()-1;it--){
      hits.push_back(*it);
    }
  } else hits=tmp;
  
  // build the transient track and fit it
  std::vector<Trajectory> trajVec;
  reco::TransientTrack theTT(theT, thePropagator->magneticField() );
  TrajectoryStateOnSurface firstState=thePropagator->propagate(theTT.impactPointState(), hits.front()->det()->surface());
  AlgebraicSymMatrix C(5,1);
  C *= 100.;
  if(!firstState.isValid()) return trajVec;
  TrajectoryStateOnSurface theTSOS( firstState.localParameters(), LocalTrajectoryError(C),
				    firstState.surface(), thePropagator->magneticField()); 
  LogDebug("SimpleTrackRefitter") << "initial TSOS\n" << theTSOS << "\n";
  const TrajectorySeed seed;//empty seed: not needed
  trajVec = theFitter->fit(seed, hits, theTSOS);
  LogDebug("SimpleTrackRefitter") <<" FITTER FOUND "<< trajVec.size() << " TRAJECTORIES" <<"\n";
  return trajVec;
}

std::vector<Trajectory> SimpleTrackRefitter::refitTrack(const TrajectorySeed& seed,
                                                   const TrackingRecHitCollection &hits,
                                                   const uint32_t ExcludedDetid)
{
  initServices(seed.direction()==alongMomentum);
  Trajectory traj=createStartingTrajectory(seed);
  TransientTrackingRecHit::RecHitContainer trans_hits;
  for (unsigned int icosmhit=hits.size()-1;icosmhit+1>0;icosmhit--){
    TransientTrackingRecHit::RecHitPointer tmphit;
    const TrackingRecHit* rh = &(hits[icosmhit]);
    if(rh->geographicalId().rawId() == ExcludedDetid) {
      InvalidTrackingRecHit* irh = new InvalidTrackingRecHit(rh->geographicalId(), TrackingRecHit::inactive);
      tmphit=RHBuilder->build(irh);
      delete irh;
    }
    else {
      tmphit=RHBuilder->build(rh);
    }
    trans_hits.push_back(&(*tmphit));
    if (icosmhit<hits.size()-1){
      TSOS prSt = thePropagator->propagate(traj.lastMeasurement().updatedState(),
                                           tracker->idToDet(hits[icosmhit].geographicalId())->surface());
      if(prSt.isValid() && tmphit->isValid()) {
       if(theUpdator->update(prSt,*tmphit).isValid()) {
	  traj.push(TrajectoryMeasurement(prSt, theUpdator->update(prSt,*tmphit),
					  tmphit, theEstimator->estimate(prSt,*tmphit).second));
       }
      }
    }
  }
  std::vector<Trajectory> smoothtraj;
  if (thePropagatorOp->propagate(traj.lastMeasurement().updatedState(),
				 tracker->idToDet((*trans_hits.begin())->geographicalId())->surface()).isValid()){
    TSOS startingState=  TrajectoryStateWithArbitraryError()
      (thePropagatorOp->propagate(traj.lastMeasurement().updatedState(),
				  tracker->idToDet((*trans_hits.begin())->geographicalId())->surface()));
    std::vector<Trajectory> fittraj=theFitter->fit(seed,trans_hits,startingState);
    if (fittraj.size()) smoothtraj=theSmoother->trajectories(*(fittraj.begin()));
  }
  destroyServices();
  return smoothtraj;
}

std::vector<Trajectory> SimpleTrackRefitter::refitTrack(const TrajectorySeed& seed,
                                                   const reco::Track& theT,
                                                   const uint32_t ExcludedDetid)
{
  initServices(seed.direction()==alongMomentum);
  Trajectory traj=createStartingTrajectory(seed);
  TransientTrackingRecHit::RecHitContainer trans_hits;
  for (trackingRecHit_iterator i=theT.recHitsBegin();i!=theT.recHitsEnd(); i++) {
    TransientTrackingRecHit::RecHitPointer tmphit;
    if((*i)->geographicalId().rawId() == ExcludedDetid) {
      InvalidTrackingRecHit* irh = new InvalidTrackingRecHit((*i)->geographicalId(), TrackingRecHit::inactive);
      tmphit=RHBuilder->build(irh);
      delete irh;
    }
    else {
      tmphit=RHBuilder->build(&**i);
    }
    trans_hits.push_back(&(*tmphit));
    if (i!=theT.recHitsBegin()){
      TSOS prSt = thePropagator->propagate(traj.lastMeasurement().updatedState(),
                                           tracker->idToDet((*i)->geographicalId())->surface());
      if(prSt.isValid() && tmphit->isValid()) {
       if(theUpdator->update(prSt,*tmphit).isValid()) {
	  traj.push(TrajectoryMeasurement(prSt, theUpdator->update(prSt,*tmphit),
					  tmphit, theEstimator->estimate(prSt,*tmphit).second));
       }
      }
    }
  }
  std::vector<Trajectory> smoothtraj;
  if (thePropagatorOp->propagate(traj.lastMeasurement().updatedState(),
				 tracker->idToDet((*trans_hits.begin())->geographicalId())->surface()).isValid()){
    TSOS startingState=  TrajectoryStateWithArbitraryError()
      (thePropagatorOp->propagate(traj.lastMeasurement().updatedState(),
				  tracker->idToDet((*trans_hits.begin())->geographicalId())->surface()));
    std::vector<Trajectory> fittraj=theFitter->fit(seed,trans_hits,startingState);
    if (fittraj.size()) smoothtraj=theSmoother->trajectories(*(fittraj.begin()));
  }
  destroyServices();
  return smoothtraj;
}

TrajectoryStateOnSurface SimpleTrackRefitter::startingTSOS(const TrajectorySeed& seed) const
{
  PTrajectoryStateOnDet pState( seed.startingState());
  const GeomDet* gdet  = tracker->idToDet(DetId(pState.detId()));
  TSOS  State= tsTransform.transientState( pState, &(gdet->surface()), &(*magfield));
  return State;

}

Trajectory SimpleTrackRefitter::createStartingTrajectory(const TrajectorySeed& seed) const
{
  Trajectory result( seed, seed.direction());
  std::vector<TrajectoryMeasurement> seedMeas = seedMeasurements(seed);
  if ( !seedMeas.empty()) {
    for (std::vector<TrajectoryMeasurement>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
      result.push(*i);
    }
  }
  return result;
}

std::vector<TrajectoryMeasurement> SimpleTrackRefitter::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; ihit != hitRange.second; ihit++) {
    TransientTrackingRecHit::RecHitPointer recHit = RHBuilder->build(&(*ihit));
    const GeomDet* hitGeomDet = tracker->idToDet( ihit->geographicalId());
    TSOS invalidState( new BasicSingleTrajectoryState( hitGeomDet->surface()));
    if (ihit == hitRange.second - 1) {
      TSOS  updatedState=startingTSOS(seed);
      result.push_back(TrajectoryMeasurement( invalidState, updatedState, recHit));
    } 
    else {
      result.push_back(TrajectoryMeasurement( invalidState, recHit));
    }
  }
  return result;
}

