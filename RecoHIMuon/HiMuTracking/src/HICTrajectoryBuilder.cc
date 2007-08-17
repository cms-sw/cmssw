#include "RecoHIMuon/HiMuTracking/interface/HICTrajectoryBuilder.h"
#include "RecoHIMuon/HiMuTracking/interface/HICTrajectoryCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"


#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "RecoTracker/CkfPattern/interface/TrajCandLess.h"
#include "RecoTracker/CkfPattern/interface/MinPtTrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/MaxHitsTrajectoryFilter.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"
#include "RecoHIMuon/HiMuSeed/interface/DiMuonTrajectorySeed.h"

using namespace std;

HICTrajectoryBuilder::
  HICTrajectoryBuilder(const edm::ParameterSet&              conf,
		       const TrajectoryStateUpdator*         updator,
		       const Propagator*                     propagatorAlong,
		       const Propagator*                     propagatorOpposite,
		       const Chi2MeasurementEstimatorBase*   estimator,
		       const TransientTrackingRecHitBuilder* RecHitBuilder,
		       const MeasurementTracker*             measurementTracker):

    theUpdator(updator),thePropagatorAlong(propagatorAlong),
    thePropagatorOpposite(propagatorOpposite),theEstimator(estimator),
    theTTRHBuilder(RecHitBuilder),theMeasurementTracker(measurementTracker),
    theLayerMeasurements(new LayerMeasurements(theMeasurementTracker)),
    theForwardPropagator(0), theBackwardPropagator(0),
    theMinPtCondition(new MinPtTrajectoryFilter(conf.getParameter<double>("ptCut"))),
    theMaxHitsCondition(new MaxHitsTrajectoryFilter(conf.getParameter<int>("maxNumberOfHits")))
{
  theMaxCand              = conf.getParameter<int>("maxCand");
  theMaxLostHit           = conf.getParameter<int>("maxLostHit");
  theMaxConsecLostHit     = conf.getParameter<int>("maxConsecLostHit");
  theLostHitPenalty       = conf.getParameter<double>("lostHitPenalty");
  theIntermediateCleaning = conf.getParameter<bool>("intermediateCleaning");
  theMinimumNumberOfHits  = conf.getParameter<int>("minimumNumberOfHits");
  theAlwaysUseInvalidHits = conf.getParameter<bool>("alwaysUseInvalidHits");
  
  cout<<" HICTrajectoryBuilder::contructor "<<endl; 
}

HICTrajectoryBuilder::~HICTrajectoryBuilder()
{
  delete theLayerMeasurements;
  delete theMinPtCondition;
  delete theMaxHitsCondition;
}

void HICTrajectoryBuilder::setEvent(const edm::Event& event) const
{
  theMeasurementTracker->update(event);
}

HICTrajectoryBuilder::TrajectoryContainer 
HICTrajectoryBuilder::trajectories(const TrajectorySeed& seed) const
{ 
 
   cout<<" HICTrajectoryBuilder::trajectories start "<<endl;
   
  TrajectoryContainer result;
  

  // analyseSeed( seed);

  TempTrajectory startingTraj = createStartingTrajectory( seed );
  
  cout<<" HICTrajectoryBuilder::trajectories starting trajectories created "<<endl;
  
//  return result;

  /// limitedCandidates( startingTraj, regionalCondition, result);
  /// FIXME: restore regionalCondition

  limitedCandidates( startingTraj, result);
  
   cout<<" HICTrajectoryBuilder::trajectories candidates found "<<result.size()<<endl;

  // analyseResult(result);

  return result;
}

TempTrajectory HICTrajectoryBuilder::
createStartingTrajectory( const TrajectorySeed& seed) const
{

  cout<<" HICTrajectoryBuilder::createStartingTrajectory "<<endl;
  
  TempTrajectory result( seed, seed.direction());
  if (  seed.direction() == alongMomentum) {
    theForwardPropagator = &(*thePropagatorAlong);
    theBackwardPropagator = &(*thePropagatorOpposite);
  }
  else {
    theForwardPropagator = &(*thePropagatorOpposite);
    theBackwardPropagator = &(*thePropagatorAlong);
  }



  std::vector<TM> seedMeas = seedMeasurements(seed);

//  std::vector<TM> seedMeas = dynamic_cast<const DiMuonTrajectorySeed*>(&seed)->measurements();
  
  std::cout<<" Size of seed "<<seedMeas.size()<<endl;
  
  if ( !seedMeas.empty()) {
  std::cout<<" TempTrajectory "<<std::endl;
    for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
  std::cout<<" TempTrajectory::before add "<<std::endl;
  std::cout<<" TempTrajectory::before estimate "<<(*i).estimate()<<std::endl;
    
      result.push(*i); 
      
  std::cout<<" TempTrajectory::after add "<<std::endl;
                 
    }
  }
   
  std::cout<<" TempTrajectory::return result "<<std::endl; 
    
  return result;
  
}

void HICTrajectoryBuilder::
limitedCandidates( TempTrajectory& startingTraj, 
		   TrajectoryContainer& result) const
{
  TempTrajectoryContainer candidates; // = TrajectoryContainer();
  TempTrajectoryContainer newCand; // = TrajectoryContainer();
  candidates.push_back( startingTraj);
// Add the additional stuff
  cout<<" HICTrajectoryBuilder::limitedCandidates "<<candidates.size()<<endl;
   
//  int theIniSign = (int)startingTraj.lastMeasurement().updatedState().freeTrajectoryState()->charge();

  int theIniSign = 1;
  dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->setSign(theIniSign);

  cout<<" Number of measurements "<<startingTraj.measurements().size()<<endl;

  while ( !candidates.empty()) {

  cout<<" HICTrajectoryBuilder::limitedCandidates::cycle "<<candidates.size()<<endl;
    newCand.clear();
    for (TempTrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
	 
	 cout<< " Before findCompatibleMeasurements "<<endl;
      std::vector<TM> meas = findCompatibleMeasurements(*traj);
	 cout<< " After findCompatibleMeasurements "<<meas.size()<<endl;

      // --- method for debugging
      if(!analyzeMeasurementsDebugger(*traj,meas,
				      theMeasurementTracker,
				      theForwardPropagator,theEstimator,
				      theTTRHBuilder)) return;
      // ---

      if ( meas.empty()) {
	if ( qualityFilter( *traj)) addToResult( *traj, result);
      }
      else {
	std::vector<TM>::const_iterator last;
	if ( theAlwaysUseInvalidHits) last = meas.end();
	else {
	  if (meas.front().recHit()->isValid()) {
	    last = find_if( meas.begin(), meas.end(), RecHitIsInvalid());
	  }
	  else last = meas.end();
	}

	for( std::vector<TM>::const_iterator itm = meas.begin(); 
	     itm != last; itm++) {
	  TempTrajectory newTraj = *traj;
	  updateTrajectory( newTraj, *itm);

	  if ( toBeContinued(newTraj)) {
	    newCand.push_back(newTraj);
	  }
	  else {
	    if ( qualityFilter(newTraj)) addToResult( newTraj, result);
	    //// don't know yet
	  }
	}
      }
    
      if ((int)newCand.size() > theMaxCand) {
	sort( newCand.begin(), newCand.end(), TrajCandLess<TempTrajectory>(theLostHitPenalty));
	newCand.erase( newCand.begin()+theMaxCand, newCand.end());
      }
    }


    if (theIntermediateCleaning) {
        candidates.clear();
        candidates = IntermediateTrajectoryCleaner::clean(newCand);
    } else {
        //cout << "calling candidates.swap(newCand) " << endl;
        candidates.swap(newCand);
    }
  }
}



#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

std::vector<TrajectoryMeasurement> 
HICTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
  TrajectoryStateTransform tsTransform;

  TrajectorySeed::range hitRange = seed.recHits();
  
  cout<<" HICTrajectoryBuilder::seedMeasurements"<<endl;
  
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
                                      ihit != hitRange.second; ihit++) {
       
       cout<<" HICTrajectoryBuilder::seedMeasurements::RecHit "<<endl;
       
    //RC TransientTrackingRecHit* recHit = TTRHbuilder->build(&(*ihit));
    TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&(*ihit));
    
    
    const GeomDet* hitGeomDet = 
      theMeasurementTracker->geomTracker()->idToDet( ihit->geographicalId());

    const DetLayer* hitLayer = 
      theMeasurementTracker->geometricSearchTracker()->detLayer(ihit->geographicalId());

    TSOS invalidState( new BasicSingleTrajectoryState( hitGeomDet->surface()));
    
    cout<<" Before ihit == hitRange.second "<<endl;
    
    if (ihit == hitRange.second - 1) {
    
    cout<<" Inside ihit == hitRange.second "<<endl;
      // the seed trajectory state should correspond to this hit
      PTrajectoryStateOnDet pState( seed.startingState());
      
      
      
      const GeomDet* gdet = theMeasurementTracker->geomTracker()->idToDet( DetId(pState.detId()));
      if (&gdet->surface() != &hitGeomDet->surface()) {
	edm::LogError("CkfPattern") << "HICTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit";
	return std::vector<TrajectoryMeasurement>(); // FIXME: should throw exception
      }

      TSOS updatedState = tsTransform.transientState( pState, &(gdet->surface()), 
						      theForwardPropagator->magneticField());
      result.push_back(TM( invalidState, updatedState, recHit, 0, hitLayer));
    }
    else {
      //----------- just a test to make the Smoother to work -----------
      
      cout<<" Outside ihit == hitRange.second "<<endl;
      
      PTrajectoryStateOnDet pState( seed.startingState());
      TSOS outerState = tsTransform.transientState( pState, &(hitGeomDet->surface()), 
						    theForwardPropagator->magneticField());
      TSOS innerState   = theBackwardPropagator->propagate(outerState,hitGeomDet->surface());
      TSOS innerUpdated = theUpdator->update(innerState,*recHit);

      result.push_back(TM( invalidState, innerUpdated, recHit, 0, hitLayer));
      //-------------------------------------------------------------

      //result.push_back(TM( invalidState, recHit, 0, hitLayer));
    }
  }

  // method for debugging
  fillSeedHistoDebugger(result.begin(),result.end());

  return result;
}

 bool HICTrajectoryBuilder::qualityFilter( const TempTrajectory& traj) const
{

//    cout << "qualityFilter called for trajectory with " 
//         << traj.foundHits() << " found hits and Chi2 = "
//         << traj.chiSquared() << endl;

  if ( traj.foundHits() >= theMinimumNumberOfHits) {
    return true;
  }
  else {
    return false;
  }
}


void HICTrajectoryBuilder::addToResult( TempTrajectory& tmptraj, 
					TrajectoryContainer& result) const
{
  Trajectory traj = tmptraj.toTrajectory();
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj);
}

void HICTrajectoryBuilder::updateTrajectory( TempTrajectory& traj,
					     const TM& tm) const
{
  TSOS predictedState = tm.predictedState();
  TM::ConstRecHitPointer hit = tm.recHit();
 
  if ( hit->isValid()) {
    TM tmp = TM( predictedState, theUpdator->update( predictedState, *hit),
		 hit, tm.estimate(), tm.layer()); 
    traj.push(tmp );
  }
  else {
    traj.push( TM( predictedState, hit, 0, tm.layer()));
  }
}

bool HICTrajectoryBuilder::toBeContinued (const TempTrajectory& traj) const
{
  if ( traj.lostHits() > theMaxLostHit) return false;

  // check for conscutive lost hits only at the end 
  // (before the last valid hit),
  // since if there was an unacceptable gap before the last 
  // valid hit the trajectory would have been stopped already

  int consecLostHit = 0;

  const TempTrajectory::DataContainer & tms = traj.measurements();
  //for( TempTrajectory::DataContainer::const_iterator itm=tms.end()-1; itm>=tms.begin(); itm--) {
  for( TempTrajectory::DataContainer::const_iterator itm=tms.rbegin(), itb = tms.rend(); itm != itb; --itm) {
    if (itm->recHit()->isValid()) break;
    else if ( // FIXME: restore this:   !Trajectory::inactive(itm->recHit()->det()) &&
	     Trajectory::lost(*itm->recHit())) consecLostHit++;
  }
  if (consecLostHit > theMaxConsecLostHit) return false; 

  // stopping condition from region has highest priority
  // if ( regionalCondition && !(*regionalCondition)(traj) )  return false;
  // next: pt-cut
  if ( !(*theMinPtCondition)(traj) )  return false;
  if ( !(*theMaxHitsCondition)(traj) )  return false;
  // finally: configurable condition
  // FIXME: restore this:  if ( !(*theConfigurableCondition)(traj) )  return false;

  return true;
}

std::vector<TrajectoryMeasurement> 
HICTrajectoryBuilder::findCompatibleMeasurements( const TempTrajectory& traj) const
{
  vector<TM> result;
  int invalidHits = 0;
  int theLowMult = 1; 

  TSOS currentState( traj.lastMeasurement().updatedState());



  vector<const DetLayer*> nl = 
                               traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());

  std::cout<<" Number of layers "<<nl.size()<<std::endl;
  
  if (nl.empty()) return result;

  int seedLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                              getDetectorCode(traj.measurements().front().layer());

  std::cout<<"findCompatibleMeasurements Point 0 "<<seedLayerCode<<std::endl;
							      
  int currentLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                               getDetectorCode(traj.lastLayer()); 
  std::cout<<"findCompatibleMeasurements Point 1 "<<currentLayerCode<<std::endl;

  for (vector<const DetLayer*>::iterator il = nl.begin(); 
                                         il != nl.end(); il++) {

   int nextLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                               getDetectorCode((*il)); 

    std::cout<<"findCompatibleMeasurements Point 2 "<<nextLayerCode<<std::endl;

   if( traj.lastLayer()->location() == GeomDetEnumerators::endcap && (**il).location() == GeomDetEnumerators::barrel )
   {
   if( abs(seedLayerCode) > 100 && abs(seedLayerCode) < 108 )
   {
      if( (**il).subDetector() == GeomDetEnumerators::PixelEndcap ) continue;
   }
   else
   {
    if(theLowMult == 0 )
    {      
      if( nextLayerCode == 0 ) continue;
    }         
      if( (**il).subDetector() == GeomDetEnumerators::TID || (**il).subDetector() == GeomDetEnumerators::TEC) continue;
   }
   }
   
   if( currentLayerCode == 101 && nextLayerCode < 100 ) {
     continue; 
   } 
   
  std::cout<<" findCompatibleMeasurements Point 3 "<<nextLayerCode<<std::endl;
   
     								       
  Trajectory traj0 = traj.toTrajectory();
  
  vector<double> theCut = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->setCuts(traj0,(*il));
  
  std::cout<<" findCompatibleMeasurements Point 4 "<<theCut[0]<<" "<<theCut[1]<<std::endl;
  
  
  // Choose Win
  int icut = 1;
  dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->chooseCuts(icut);
       
  std::cout<<" findCompatibleMeasurements Point 5 "<<theCut[0]<<" "<<theCut[1]<<std::endl;

    vector<TM> tmp0 = 
      theLayerMeasurements->measurements((**il),currentState, *theForwardPropagator, *theEstimator);
      
  std::cout<<" findCompatibleMeasurements Point 6 "<<theCut[0]<<" "<<theCut[1]<<std::endl;
  
  std::cout<<" findCompatibleMeasurements Point 7 "<<traj0.measurements().size()<<std::endl;

//   
// ========================= Choose Cut and filter =================================
//
  vector<TM> tmp;
  if( traj0.measurements().size() == 1 )
  {
     icut = 2;
     dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->chooseCuts(icut);
     std::cout<<" findCompatibleMeasurements Point 7 "<<theCut[0]<<" "<<theCut[1]<<std::endl;
     const MagneticField * mf = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->getField();
      HICTrajectoryCorrector* theCorrector = new HICTrajectoryCorrector(mf);
     std::cout<<" findCompatibleMeasurements Point 8 "<<theCut[0]<<" "<<theCut[1]<<std::endl;
    
     
     for( vector<TM>::iterator itm = tmp0.begin(); itm != tmp0.end(); itm++ )
     {
        TM tm = (*itm);
        TSOS predictedState = tm.predictedState();
	TM::ConstRecHitPointer  hit = tm.recHit();
	TSOS updateState = traj0.lastMeasurement().updatedState();
	
	std::cout<<" findCompatibleMeasurements Point 9 "<<theCut[0]<<" "<<theCut[1]<<std::endl;
	
        TSOS predictedState0 = theCorrector->correct( (*traj0.lastMeasurement().updatedState().freeTrajectoryState()), 
                                                      (*(predictedState.freeTrajectoryState())), 
			                              hit->det() );
						      
						      
	std::cout<<"findCompatibleMeasurements  Point 10 "<<theCut[0]<<" "<<theCut[1]<<std::endl;
						      
        if(predictedState0.isValid()) predictedState = predictedState0;
	if((*hit).isValid())
	{
  
              double accept=
              (dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->estimate(predictedState,*hit)).first; 
	      
	std::cout<<" findCompatibleMeasurements Point 11 "<<accept<<std::endl;
	      
              if(!accept) {
	std::cout<<" findCompatibleMeasurements Point 11.1 "<<accept<<std::endl;
	      
	      continue;
	      }
              tmp.push_back(TM(predictedState, updateState, hit, tm.estimate(), tm.layer()));
     
	std::cout<<" findCompatibleMeasurements  Point 12 "<<std::endl;
	      
	}
     }					   
     delete theCorrector;	
     std::cout<<" findCompatibleMeasurements  Point 13 "<<std::endl;
					   
  }
     else
     {
     std::cout<<" findCompatibleMeasurements  Point 14 "<<std::endl;
        tmp = tmp0;
        
     }
//        tmp = tmp0;

     std::cout<<" findCompatibleMeasurements  Point 15 "<<std::endl;



    if ( !tmp.empty()) {
      if ( result.empty()) result = tmp;
      else {
	// keep one dummy TM at the end, skip the others
	result.insert( result.end()-invalidHits, tmp.begin(), tmp.end());
      }
      invalidHits++;
    }
     std::cout<<" Point 16 "<<std::endl;
  }

  // sort the final result, keep dummy measurements at the end
  if ( result.size() > 1) {
    sort( result.begin(), result.end()-invalidHits, TrajMeasLessEstim());
  }
     std::cout<<" Point 17 "<<std::endl;

#ifdef DEBUG_INVALID
  bool afterInvalid = false;
  for (vector<TM>::const_iterator i=result.begin();
       i!=result.end(); i++) {
    if ( ! i->recHit().isValid()) afterInvalid = true;
    if (afterInvalid && i->recHit().isValid()) {
      edm::LogError("CkfPattern") << "HICTrajectoryBuilder error: valid hit after invalid!" ;
    }
  }
#endif

  //analyseMeasurements( result, traj);

  return result;
}

