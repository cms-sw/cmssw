#include "RecoTracker/CkfPattern/interface/CombinatorialTrajectoryBuilder.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "RecoTracker/CkfPattern/interface/TrajCandLess.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
//#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

CombinatorialTrajectoryBuilder::
CombinatorialTrajectoryBuilder(const edm::ParameterSet& conf){
  //minimum number of hits per tracks
  //theMinHits=conf.getParameter<int>("MinHits");
  //cut on chi2
  chi2cut=conf.getParameter<double>("Chi2Cut");
}


void CombinatorialTrajectoryBuilder::init(const edm::EventSetup& es)
{     
  //services
  es.get<IdealMagneticFieldRecord>().get(magfield);
  es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );

  //trackingtools
  //thePropagator=        new AnalyticalPropagator(&(*magfield), alongMomentum);
  thePropagator=        new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield));
  theUpdator=           new KFUpdator();
  theEstimator=         new Chi2MeasurementEstimator(chi2cut);
  theNavigationSchool=  new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*magfield));
  theMeasurementTracker=new MeasurementTracker(es);
  theLayerMeasurements= new LayerMeasurements(theMeasurementTracker);


  theMaxCand = 5;
  theMaxLostHit = 3;
  theMaxConsecLostHit = 2;
  theLostHitPenalty = 30 ;
  theMinHits = 5; 
  theAlwaysUseInvalid = true;
}




CombinatorialTrajectoryBuilder::TrajectoryContainer 
CombinatorialTrajectoryBuilder::trajectories(const TrajectorySeed& seed,edm::Event& e)
{
  // update theMeasDetTracker
  theMeasurementTracker->update(e);

  // set the correct navigation
  NavigationSetter setter( *theNavigationSchool);
  
  // set the propagation direction
  thePropagator->setPropagationDirection(seed.direction());

  TrajectoryContainer result;

  // analyseSeed( seed);

  //cout << "--- calling createStartingTrajectory" << endl;
  Trajectory startingTraj = createStartingTrajectory( seed);

  /// limitedCandidates( startingTraj, regionalCondition, result);
  /// FIXME: restore regionalCondition
  //cout << "--- calling limitedCandidates" << endl;
  limitedCandidates( startingTraj, result);

  // analyseResult(result);

  return result;
}

Trajectory CombinatorialTrajectoryBuilder::
createStartingTrajectory( const TrajectorySeed& seed) const
{
  Trajectory result( seed, seed.direction());

  std::vector<TM> seedMeas = seedMeasurements(seed);
  if ( !seedMeas.empty()) {
    for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
      result.push(*i);
      
      /*   // DEBUG INFO
      cout << "i->recHit()->globalPosition().perp(): " 
	   <<  i->recHit()->globalPosition().perp() << endl;
      cout << "i->recHit()->globalPosition().eta(): " 
	   <<  i->recHit()->globalPosition().eta() << endl;
      cout << "i->recHit()->globalPosition().phi(): " 
	   <<  i->recHit()->globalPosition().phi() << endl;
      cout << "i->recHit()->globalPosition().z(): " 
	   <<  i->recHit()->globalPosition().z() << endl;
      */
      
    }
  }
  return result;
}

void CombinatorialTrajectoryBuilder::
limitedCandidates( Trajectory& startingTraj, 
		   TrajectoryContainer& result)
{
  TrajectoryContainer candidates = TrajectoryContainer();
  TrajectoryContainer newCand = TrajectoryContainer();
  candidates.push_back( startingTraj);

  while ( !candidates.empty()) {

    newCand.clear();
    for (TrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
      std::vector<TM> meas = findCompatibleMeasurements(*traj);
      if ( meas.empty()) {
	if ( qualityFilter( *traj)) addToResult( *traj, result);
      }
      else {
	std::vector<TM>::const_iterator last;
	if ( theAlwaysUseInvalid) last = meas.end();
	else {
	  if (meas.front().recHit()->isValid()) {
	    last = find_if( meas.begin(), meas.end(), RecHitIsInvalid());
	  }
	  else last = meas.end();
	}

	for( std::vector<TM>::const_iterator itm = meas.begin(); 
	     itm != last; itm++) {
	  Trajectory newTraj = *traj;
	  updateTrajectory( newTraj, *itm);

	  if ( toBeContinued(newTraj)) {
	    newCand.push_back(newTraj);
	    cout << "newCand.size(): " << newCand.size() << endl;
	  }
	  else {
	    if ( qualityFilter(newTraj)) addToResult( newTraj, result);
	    //// don't know yet
	  }
	}
      }
    
      if ((int)newCand.size() > theMaxCand) {
	sort( newCand.begin(), newCand.end(), TrajCandLess(theLostHitPenalty));
	newCand.erase( newCand.begin()+theMaxCand, newCand.end());
      }
    }


    // FIXME: restore intermediary cleaning
    //if (theIntermediateCleaning) {
    // candidates.clear();
    // candidates = intermediaryClean(newCand);
    //} else {
    //cout << "calling candidates.swap(newCand) " << endl;
    candidates.swap(newCand);
      //}
  }
}



#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

std::vector<TrajectoryMeasurement> 
CombinatorialTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
  TkTransientTrackingRecHitBuilder recHitBuilder( theMeasurementTracker->geomTracker());
  TrajectoryStateTransform tsTransform;

  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
       ihit != hitRange.second; ihit++) {
    TransientTrackingRecHit* recHit = recHitBuilder.build(&(*ihit));
    const GeomDet* hitGeomDet = 
      theMeasurementTracker->geomTracker()->idToDet( ihit->geographicalId());

    const DetLayer* hitLayer = 
      theMeasurementTracker->geometricSearchTracker()->detLayer(ihit->geographicalId());

    TSOS invalidState( new BasicSingleTrajectoryState( hitGeomDet->surface()));
    if (ihit == hitRange.second - 1) {
      // the seed trajectory state should correspond to this hit
      PTrajectoryStateOnDet pState( seed.startingState());
      const GeomDet* gdet = theMeasurementTracker->geomTracker()->idToDet( DetId(pState.detId()));
      if (&gdet->surface() != &hitGeomDet->surface()) {
	std::cout << "CombinatorialTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit" 
		  << std::endl;
	return std::vector<TrajectoryMeasurement>(); // FIXME: should throw exception
      }

      TSOS updatedState = tsTransform.transientState( pState, &(gdet->surface()), 
						      thePropagator->magneticField());
      result.push_back(TM( invalidState, updatedState, recHit, 0, hitLayer));
    }
    else {
      result.push_back(TM( invalidState, recHit, 0, hitLayer));
    }
  }
  return result;
}

 bool CombinatorialTrajectoryBuilder::qualityFilter( const Trajectory& traj)
{

//    cout << "qualityFilter called for trajectory with " 
//         << traj.foundHits() << " found hits and Chi2 = "
//         << traj.chiSquared() << endl;

  if ( traj.foundHits() >= theMinHits) {
    return true;
  }
  else {
    return false;
  }
}


void CombinatorialTrajectoryBuilder::addToResult( Trajectory& traj, 
						  TrajectoryContainer& result)
{
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj);
}

void CombinatorialTrajectoryBuilder::updateTrajectory( Trajectory& traj,
						       const TM& tm) const
{
  TSOS predictedState = tm.predictedState();
  const TransientTrackingRecHit* hit = tm.recHit();
 
  /* DEBUG INFO
  cout << "before updateTraject: " << endl;
  cout << "tm.predictedState().globalPosition().perp(): " 
       << tm.predictedState().globalPosition().perp() << endl;
  cout << "tm.predictedState().globalPosition().eta(): " 
       << tm.predictedState().globalPosition().eta() << endl;
  cout << "tm.predictedState().globalPosition().phi(): " 
       << tm.predictedState().globalPosition().phi() << endl;
  cout << "tm.predictedState().globalPosition().z(): " 
       << tm.predictedState().globalPosition().z() << endl;

  cout << "tm.predictedState().globalMomentum().perp(): " 
       << tm.predictedState().globalMomentum().perp() << endl;
  cout << "tm.predictedState().globalMomentum().eta(): " 
       << tm.predictedState().globalMomentum().eta() << endl;
  cout << "tm.predictedState().globalMomentum().phi(): " 
       << tm.predictedState().globalMomentum().phi() << endl;
  cout << "tm.predictedState().globalMomentum().z(): " 
       << tm.predictedState().globalMomentum().z() << endl;
  */

  if ( hit->isValid()) {
    TM tmp = TM( predictedState, theUpdator->update( predictedState, *hit),
		 hit, tm.estimate(), tm.layer()); 

    //cout << "updatedState.globalMomentum().perp(): " 
    // << tmp.updatedState().globalMomentum().perp() << endl;

    traj.push(tmp );
  }
  else {
    traj.push( TM( predictedState, hit, 0, tm.layer()));
  }
}

bool CombinatorialTrajectoryBuilder::toBeContinued (const Trajectory& traj)
{
  if ( traj.lostHits() > theMaxLostHit) return false;

  // check for conscutive lost hits only at the end 
  // (before the last valid hit),
  // since if there was an unacceptable gap before the last 
  // valid hit the trajectory would have been stopped already

  int consecLostHit = 0;
  vector<TM> tms = traj.measurements();
  for( vector<TM>::const_iterator itm=tms.end()-1; itm>=tms.begin(); itm--) {
    if (itm->recHit()->isValid()) break;
    else if ( // FIXME: restore this:   !Trajectory::inactive(itm->recHit()->det()) &&
	      Trajectory::lost(*itm->recHit())) consecLostHit++;
  }
  if (consecLostHit > theMaxConsecLostHit) return false; 

  // stopping condition from region has highest priority
  // if ( regionalCondition && !(*regionalCondition)(traj) )  return false;
  // next: pt-cut
  // FIXME: restore this:  if ( !(*theMinPtCondition)(traj) )  return false;
  // finally: configurable condition
  // FIXME: restore this:  if ( !(*theConfigurableCondition)(traj) )  return false;

  return true;
}

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

std::vector<TrajectoryMeasurement> 
CombinatorialTrajectoryBuilder::findCompatibleMeasurements( const Trajectory& traj){
  cout << "start findCompMeas" << endl;
  cout << "traj.foundHits(): " << traj.foundHits() << endl;
  cout << "traj.lostHits(): " << traj.lostHits() << endl;
  const BarrelDetLayer* barrelPointer = dynamic_cast<const BarrelDetLayer*> (traj.lastLayer());
  if(barrelPointer) 
    cout << "lastLayer.specificSurface().radius(): "
	 << barrelPointer->specificSurface().radius() << endl;

  TrajectoryStateOnSurface testState = traj.lastMeasurement().forwardPredictedState();

  /*  ------- DEBUG INFO ------------
  if( traj.lastMeasurement().recHit()->isValid() ) {
    AlgebraicVector parTS = traj.lastMeasurement().recHit()->parameters(testState) ;
    AlgebraicVector par   = traj.lastMeasurement().recHit()->parameters() ;

    cout << "parTS: " << parTS << endl;
    cout << "par:   " << par << endl;

    cout << "lastHit is valid and :" << endl;
    cout << "traj.lastMeasurement().recHit().globalPosition().perp(): " 
	 << traj.lastMeasurement().recHit()->globalPosition().perp() << endl;
    cout << "traj.lastMeasurement().recHit().globalPosition().eta(): " 
	 << traj.lastMeasurement().recHit()->globalPosition().eta() << endl;
    cout << "traj.lastMeasurement().recHit().globalPosition().phi(): " 
	 << traj.lastMeasurement().recHit()->globalPosition().phi() << endl;
    cout << "traj.lastMeasurement().recHit().globalPosition().z(): " 
	 << traj.lastMeasurement().recHit()->globalPosition().z() << endl;
  }

  cout << "traj.lastMeasurement().updatedState().globalMomentum().perp(): " 
       << traj.lastMeasurement().updatedState().globalMomentum().perp() << endl;
  cout << "traj.lastMeasurement().updatedState().globalMomentum().eta(): " 
       << traj.lastMeasurement().updatedState().globalMomentum().eta() << endl;
  cout << "traj.lastMeasurement().updatedState().globalMomentum().phi(): " 
       << traj.lastMeasurement().updatedState().globalMomentum().phi() << endl;
  cout << "traj.lastMeasurement().updatedState().globalMomentum().z(): " 
       << traj.lastMeasurement().updatedState().globalMomentum().z() << endl;
  */


  vector<TM> result;
  int invalidHits = 0;

  // const FTS& currFts( *traj.lastFts());
  TSOS currentState( traj.lastMeasurement().updatedState());

  vector<const DetLayer*> nl = 
    traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());

  cout << "nlayer.size(): " << nl.size() << endl;

  if (nl.empty()) return result;

  for (vector<const DetLayer*>::iterator il = nl.begin(); 
       il != nl.end(); il++) {
    vector<TM> tmpp = 
      theLayerMeasurements->measurements((**il),currentState, *thePropagator, *theEstimator);

    vector<TM> tmp;
    for(vector<TM>::const_iterator tmpIt=tmpp.begin();tmpIt!=tmpp.end();tmpIt++){
      tmp.push_back(  TM(tmpIt->predictedState(),tmpIt->recHit(),tmpIt->estimate(),*il)  );
    }

    cout << "in findCompatibleMeasurement found " << tmp.size()<< " measurements" << endl;

    //(**il).measurements( currentState, *thePropagator, *theEstimator);
    if ( !tmp.empty()) {
      if ( result.empty()) result = tmp;
      else {
	// keep one dummy TM at the end, skip the others
	result.insert( result.end()-invalidHits, tmp.begin(), tmp.end());
      }
      invalidHits++;
    }
  }

  // sort the final result, keep dummy measurements at the end
  if ( result.size() > 1) {
    sort( result.begin(), result.end()-invalidHits, TrajMeasLessEstim());
  }

#ifdef DEBUG_INVALID
  bool afterInvalid = false;
  for (vector<TM>::const_iterator i=result.begin();
       i!=result.end(); i++) {
    if ( ! i->recHit().isValid()) afterInvalid = true;
    if (afterInvalid && i->recHit().isValid()) {
      cout << "CombinatorialTrajectoryBuilder error: valid hit avter invalid!" 
	   << endl;
    }
  }
#endif

  //analyseMeasurements( result, traj);

  return result;
}

