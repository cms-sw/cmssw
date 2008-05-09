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
#include "TrackingTools/TrajectoryFiltering/interface/MinPtTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxHitsTrajectoryFilter.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "RecoHIMuon/HiMuSeed/interface/DiMuonTrajectorySeed.h"
#include "RecoHIMuon/HiMuTracking/interface/HICMuonUpdator.h"

using namespace std;

HICTrajectoryBuilder::
  HICTrajectoryBuilder(const edm::ParameterSet&              conf,
		       const TrajectoryStateUpdator*         updator,
		       const Propagator*                     propagatorAlong,
		       const Propagator*                     propagatorOpposite,
		       const Chi2MeasurementEstimatorBase*   estimator,
		       const TransientTrackingRecHitBuilder* RecHitBuilder,
		       const MeasurementTracker*             measurementTracker,
                       const TrajectoryFilter*               filter):

    theUpdator(updator),thePropagatorAlong(propagatorAlong),
    thePropagatorOpposite(propagatorOpposite),theEstimator(estimator),
    theTTRHBuilder(RecHitBuilder),theMeasurementTracker(measurementTracker),
    theLayerMeasurements(new LayerMeasurements(theMeasurementTracker)),
    theForwardPropagator(0), theBackwardPropagator(0),
    BaseCkfTrajectoryBuilder(conf,
                             updator, propagatorAlong,propagatorOpposite,
                             estimator, RecHitBuilder, measurementTracker,filter)
{
//  theMaxCand              = conf.getParameter<int>("maxCand");
//  theMaxLostHit           = conf.getParameter<int>("maxLostHit");
//  theMaxConsecLostHit     = conf.getParameter<int>("maxConsecLostHit");
//  theLostHitPenalty       = conf.getParameter<double>("lostHitPenalty");
//  theIntermediateCleaning = conf.getParameter<bool>("intermediateCleaning");
//  theMinimumNumberOfHits  = conf.getParameter<int>("minimumNumberOfHits");
//  theAlwaysUseInvalidHits = conf.getParameter<bool>("alwaysUseInvalidHits");

  theMaxCand              = 1;
  theMaxLostHit           = 0;
  theMaxConsecLostHit     = 0;
  theLostHitPenalty       = 0.;
  theIntermediateCleaning = false;
  theMinimumNumberOfHits  = 6;
  theAlwaysUseInvalidHits = false;
  
//  cout<<" HICTrajectoryBuilder::contructor "<<endl; 
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
   theHICConst = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->getHICConst(); 
 //  cout<<" HICTrajectoryBuilder::trajectories start with seed"<<endl;
   
  TrajectoryContainer result;

  // analyseSeed( seed);

  TempTrajectory startingTraj = createStartingTrajectory( seed );
  
 // cout<<" HICTrajectoryBuilder::trajectories starting trajectories created "<<startingTraj.empty()<<endl;
  
  if(startingTraj.empty()) {
  //      cout<<" Problem with starting trajectory "<<endl; 
  return result;}

  /// limitedCandidates( startingTraj, regionalCondition, result);
  /// FIXME: restore regionalCondition

  limitedCandidates( startingTraj, result);
  
 //  cout<<" HICTrajectoryBuilder::trajectories candidates found "<<result.size()<<endl;

  // analyseResult(result);

  return result;
}

TempTrajectory HICTrajectoryBuilder::
createStartingTrajectory( const TrajectorySeed& seed) const
{

  //cout<<" HICTrajectoryBuilder::createStartingTrajectory "<<seed.direction()<<endl;
  
//  TempTrajectory result( seed, seed.direction());
  TempTrajectory result( seed, oppositeToMomentum );

//  if (  seed.direction() == alongMomentum) {
//    theForwardPropagator = &(*thePropagatorAlong);
//    theBackwardPropagator = &(*thePropagatorOpposite);
//  }
//  else {
    theForwardPropagator = &(*thePropagatorOpposite);
    theBackwardPropagator = &(*thePropagatorAlong);
//  }



  std::vector<TM> seedMeas = seedMeasurements(seed);

//  std::vector<TM> seedMeas = dynamic_cast<const DiMuonTrajectorySeed*>(&seed)->measurements();
  
//  std::cout<<" Size of seed "<<seedMeas.size()<<endl;
  
  if ( !seedMeas.empty()) {
//  std::cout<<" TempTrajectory "<<std::endl;
    for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
//  std::cout<<" TempTrajectory::before add "<<std::endl;
    
   result.push(*i); 
      
//  std::cout<<" TempTrajectory::after add "<<std::endl;
                 
    }
  }
   
//  std::cout<<" TempTrajectory::return result "<<std::endl; 
    
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
//  cout<<" HICTrajectoryBuilder::limitedCandidates "<<candidates.size()<<endl;
   
//  int theIniSign = (int)startingTraj.lastMeasurement().updatedState().freeTrajectoryState()->charge();

  int theIniSign = 1;
  dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->setSign(theIniSign);

  //cout<<" Number of measurements "<<startingTraj.measurements().size()<<endl;

 // return;

  while ( !candidates.empty()) {

 // cout<<" HICTrajectoryBuilder::limitedCandidates::cycle "<<candidates.size()<<endl;

    newCand.clear();

    for (TempTrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
	 
//	 cout<< " Before findCompatibleMeasurements "<<endl;
      std::vector<TM> meas = findCompatibleMeasurements(*traj);
//	 cout<< " After findCompatibleMeasurements "<<meas.size()<<endl;


      if ( meas.empty()) {
//        cout<<": Measurements empty : "<<endl;
	if ( qualityFilter( *traj)) addToResult( *traj, result);
      }
      else {
//        cout<<" : Update trajectoris :   "<<endl;
	for( std::vector<TM>::const_iterator itm = meas.begin(); 
                                       	     itm != meas.end(); itm++) {
	  TempTrajectory newTraj = *traj;
	  bool good = updateTrajectory( newTraj, *itm);
          if(good)
          {
	    if ( toBeContinued(newTraj)) {
//               cout<<": toBeContinued :"<<endl;
	       newCand.push_back(newTraj);
	     }
	     else {
//               cout<<": good TM : to be stored :"<<endl;
	       if ( qualityFilter(newTraj)) addToResult( newTraj, result);
	    //// don't know yet
	          }
          } // good
            else
            {
//                  cout<<": bad TM : to be stored :"<<endl;
               if ( qualityFilter( *traj)) addToResult( *traj, result);
            }
	 } //meas 
      }
    
      if ((int)newCand.size() > theMaxCand) {
	sort( newCand.begin(), newCand.end(), TrajCandLess<TempTrajectory>(theLostHitPenalty));
	newCand.erase( newCand.begin()+theMaxCand, newCand.end());
      }
    }
        candidates.swap(newCand);
  }
}



#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoHIMuon/HiMuSeed/interface/DiMuonTrajectorySeed.h"

std::vector<TrajectoryMeasurement> 
HICTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
  TrajectoryStateTransform tsTransform;

  //TrajectorySeed::range hitRange = seed.recHits();
  
//  cout<<" HICTrajectoryBuilder::seedMeasurements number of TM "<<dynamic_cast<DiMuonTrajectorySeed*>(const_cast<TrajectorySeed*>(&seed))->measurements().size()<<endl;
  

   std::vector<TrajectoryMeasurement> start = dynamic_cast<DiMuonTrajectorySeed*>(const_cast<TrajectorySeed*>(&seed))->measurements();
   for(std::vector<TrajectoryMeasurement>::iterator imh = start.begin(); imh != start.end(); imh++)
   { 
       
   //  cout<<" HICTrajectoryBuilder::seedMeasurements::RecHit "<<endl;
       
     result.push_back(*imh);
   }

  // method for debugging
  fillSeedHistoDebugger(result.begin(),result.end());

  return result;
}

 bool HICTrajectoryBuilder::qualityFilter( const TempTrajectory& traj) const
{

  //  cout << "qualityFilter called for trajectory with " 
    //     << traj.foundHits() << " found hits and Chi2 = "
      //   << traj.chiSquared() << endl;

  if ( traj.foundHits() < theMinimumNumberOfHits) {
    return false;
  }

  Trajectory traj0 = traj.toTrajectory();
// Check the number of pixels
  const Trajectory::DataContainer tms = traj0.measurements();
  //for( TempTrajectory::DataContainer::const_iterator itm=tms.end()-1; itm>=tms.begin(); itm--) {
  int ipix = 0; 
  for( Trajectory::DataContainer::const_iterator itm = tms.begin(); itm != tms.end(); itm++) {
     if((*itm).layer()->subDetector() == GeomDetEnumerators::PixelEndcap || (*itm).layer()->subDetector() == GeomDetEnumerators::PixelBarrel) ipix++;
  }
// cout<<" Number of pixels "<<ipix<<endl;
 if(ipix < 2) return false;
//
// Refit the trajectory
//
/*
  ConstRecHitContainer myrechits = traj0.recHits();

  PTrjectoryStateOnDet garbage1;
  edm::OwnVector<TrackingRecHit> garbage2;
  PropagationDirection propDir = alongMomentum;
  TrajectorySeed seed(garbage1,garbage2,propDir);
  vector<Trajectory> trajectories = theFitter->fit(seed,recHitsForReFit,firstTSOS);
  if(trajectories.empty()) return false;
  Trajectory trajectoryBW = trajectories.front();
  vector<Trajectory> trajectoriesSM = theSmoother->trajectories(trajectoryBW);
*/

 return true;
}


void HICTrajectoryBuilder::addToResult( TempTrajectory& tmptraj, 
					TrajectoryContainer& result) const
{
  Trajectory traj = tmptraj.toTrajectory();
// Refit the trajectory before putting into the last result

//  ConstRecHitContainer myrechits = traj0.recHits();

//  PTrajectoryStateOnDet garbage1;
//  edm::OwnVector<TrackingRecHit> garbage2;
//  PropagationDirection propDir = alongMomentum;
//  TrajectoryStateOnSurface firstTSOS = *(traj0.lastMeasurement().freeTrajectoryState());
//  TrajectorySeed seed(garbage1,garbage2,propDir);

//  vector<Trajectory> trajectories = theFitter->fit(seed,recHitsForReFit,firstTSOS);
//  if(trajectories.empty()) return false;
//  Trajectory trajectoryBW = trajectories.front();
//  vector<Trajectory> trajectoriesSM = theSmoother->trajectories(trajectoryBW);

//  if( trajectoriesSM.empty())
//  {
//   Trajectory traj = trajectoriesSM.front();
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj);
//  } // trajectories SM
}

bool HICTrajectoryBuilder::updateTrajectory( TempTrajectory& traj,
					     const TM& tm) const
{
  bool good=false; 
  TSOS predictedState = tm.predictedState();
  TM::ConstRecHitPointer hit = tm.recHit();
  Trajectory traj0 = traj.toTrajectory();
// My update
   vector<double> theCut = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->setCuts(traj0,tm.layer());
   int icut = 3;
   dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->chooseCuts(icut);
   const MagneticField * mf = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->getField();
   HICMuonUpdator hicup(theCut[2],theCut[3], mf,theHICConst);
   double chi2rz,chi2rf;
 
  if ( hit->isValid()) {

// Update trajectory
//
  TrajectoryStateOnSurface newUpdateState=hicup.update(traj0, predictedState, tm, tm.layer(), chi2rz, chi2rf);
  double accept=
              (dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->estimate(newUpdateState,*hit)).first;
  if(accept)
  {
  //std::cout<<" updateTrajectory::UpdateState::New momentum "<<newUpdateState.freeTrajectoryState()->momentum().perp()<<" "<<newUpdateState.freeTrajectoryState()->momentum().z()<<std::endl;

  TM tmp = TM(predictedState, newUpdateState, hit, tm.estimate(), tm.layer());

  traj.push(tmp );
  good=true;
  return good;
  }
    else
    {
    //   std::cout<<" updateTrajectory::failed after update "<<accept<<std::endl;
       return good;
    }
  }
  else {
    return good;
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
  //if ( !(*theMinPtCondition)(traj) )  return false;
  //if ( !(*theMaxHitsCondition)(traj) )  return false;
  // finally: configurable condition
  // FIXME: restore this:  if ( !(*theConfigurableCondition)(traj) )  return false;

  return true;
}

std::vector<TrajectoryMeasurement> 
HICTrajectoryBuilder::findCompatibleMeasurements( const TempTrajectory& traj) const
{
  //cout<<" HICTrajectoryBuilder::FindCompatibleMeasurement start "<<traj.empty()<<endl; 
  vector<TM> result;
  int invalidHits = 0;
  int theLowMult = 1; 

  TSOS currentState( traj.lastMeasurement().updatedState());

  vector<const DetLayer*> nl = 
                               traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());

  //std::cout<<" Number of layers "<<nl.size()<<std::endl;
  
  if (nl.empty()) return result;

  int seedLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                              getDetectorCode(traj.measurements().front().layer());

  //std::cout<<"findCompatibleMeasurements Point 0 "<<seedLayerCode<<std::endl;
							      
  int currentLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                               getDetectorCode(traj.lastLayer()); 
 // std::cout<<"findCompatibleMeasurements Point 1 "<<currentLayerCode<<std::endl;

  for (vector<const DetLayer*>::iterator il = nl.begin(); 
                                         il != nl.end(); il++) {

   int nextLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                               getDetectorCode((*il)); 

 // std::cout<<"findCompatibleMeasurements Point 2 "<<nextLayerCode<<std::endl;

   if( traj.lastLayer()->location() == GeomDetEnumerators::endcap && (**il).location() == GeomDetEnumerators::barrel )
   {
   if( abs(seedLayerCode) > 100 && abs(seedLayerCode) < 108 )
   {
      if( (**il).subDetector() == GeomDetEnumerators::PixelEndcap ) continue;
   } // 100-108
   else
   {
    if(theLowMult == 0 )
    {      
      if( nextLayerCode == 0 ) continue;
    }         
      if( (**il).subDetector() == GeomDetEnumerators::TID || (**il).subDetector() == GeomDetEnumerators::TEC) continue;
   } // 100-108
   } // barrel and endcap
   
   if( currentLayerCode == 101 && nextLayerCode < 100 ) {
     continue; 
   }  // currentLayer-nextLayer
   
 // std::cout<<" findCompatibleMeasurements Point 3 "<<nextLayerCode<<std::endl;
   
     								       
  Trajectory traj0 = traj.toTrajectory();
  
  vector<double> theCut = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->setCuts(traj0,(*il));
  
  // Choose Win
  int icut = 1;
  dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->chooseCuts(icut);

 // std::cout<<" findCompatibleMeasurements::current state : "<<
   //          " charge "<< currentState.freeTrajectoryState()->parameters().charge()<<
     //        " pt "<<currentState.freeTrajectoryState()->parameters().momentum().perp()<<
       //      " pz "<<currentState.freeTrajectoryState()->parameters().momentum().z()<<
         //    " r  "<<currentState.freeTrajectoryState()->parameters().position().perp()<<
         //    " phi  "<<currentState.freeTrajectoryState()->parameters().position().phi()<<
         //    " z  "<<currentState.freeTrajectoryState()->parameters().position().z()<<
         //    endl; 

  vector<TM> tmp0 = 
                        theLayerMeasurements->measurements((**il), currentState, *theForwardPropagator, *theEstimator);
      
//  std::cout<<" findCompatibleMeasurements Point 6 "<<theCut[0]<<" "<<theCut[1]<<std::endl;
//  std::cout<<" findCompatibleMeasurements Point 7 "<<traj0.measurements().size()<<std::endl;

//   
// ========================= Choose Cut and filter =================================
//
     vector<TM> tmp;
     icut = 2;
     dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->chooseCuts(icut);
     const MagneticField * mf = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->getField();

  //   std::cout<<" findCompatibleMeasurements Point 8 "<<theCut[0]<<" "<<theCut[1]<<" Size of candidates "<<tmp0.size()<<std::endl;
   
     for( vector<TM>::iterator itm = tmp0.begin(); itm != tmp0.end(); itm++ )
     {
        TM tm = (*itm);
        TSOS predictedState = tm.predictedState();
	TM::ConstRecHitPointer  hit = tm.recHit();
	TSOS updateState = traj0.lastMeasurement().updatedState();
	
  if( traj0.measurements().size() == 1 )
  {
//        HICTrajectoryCorrector* theCorrector = new HICTrajectoryCorrector(mf,theHICConst);	
        HICTrajectoryCorrector theCorrector(mf,theHICConst);
        TSOS predictedState0 = theCorrector.correct( (*traj0.lastMeasurement().updatedState().freeTrajectoryState()), 
                                                      (*(predictedState.freeTrajectoryState())), 
			                              hit->det() );
						      
						      
        if(predictedState0.isValid()) {
         //     std::cout<<" Accept the corrected state "<<std::endl; 
         predictedState = predictedState0;}
	if((*hit).isValid())
	{
  
              double accept=
              (dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->estimate(predictedState,*hit)).first; 
	      
              if(!accept) {
	   //       std::cout<<" findCompatibleMeasurements::failed after the first step "<<accept<<std::endl;
	      continue;
	      } // accept
        } // Hit Valid
//        delete theCorrector;
   } // first step

          tmp.push_back(TM(predictedState, updateState, hit, tm.estimate(), tm.layer()));
  }		// tm 			   


    if ( !tmp.empty()) {
      if ( result.empty() ) result = tmp;
      else {
        // keep one dummy TM at the end, skip the others
        result.insert( result.end(), tmp.begin(), tmp.end());
      }
    }
   //  std::cout<<" Results size "<<result.size()<<std::endl;
  } // next layers

  // sort the final result, keep dummy measurements at the end
  if ( result.size() > 1) {
    sort( result.begin(), result.end(), TrajMeasLessEstim());
  }
  //analyseMeasurements( result, traj);

  return result;
}

