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
#include "RecoHIMuon/HiMuPropagator/interface/HICMuonPropagator.h"


#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
//#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"

using namespace std;
using namespace cms;

//#define DEBUG

HICTrajectoryBuilder::
  HICTrajectoryBuilder(const edm::ParameterSet&              conf, 
                       const edm::EventSetup&                es1,
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
  theMaxCand              = 1;
  theMaxLostHit           = 0;
  theMaxConsecLostHit     = 0;
  theLostHitPenalty       = 0.;
  theIntermediateCleaning = false;
  theMinimumNumberOfHits  = 5;
  theAlwaysUseInvalidHits = false;
  es1.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);
  es1.get<TrackingComponentsRecord>().get("KFFitterForRefitInsideOut",theFitterTrack);
  es1.get<TrackingComponentsRecord>().get("KFSmootherForRefitInsideOut",theSmootherTrack);
  es1.get<TrackingComponentsRecord>().get("SmartPropagatorAny",thePropagatorTrack);
  
#ifdef DEBUG  
  cout<<" HICTrajectoryBuilder::contructor "<<endl;
#endif 
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
#ifdef DEBUG
   cout<<" HICTrajectoryBuilder::trajectories start with seed"<<endl;
#endif   
  TrajectoryContainer result;

  TempTrajectory startingTraj = createStartingTrajectory( seed );
#ifdef DEBUG  
  cout<<" HICTrajectoryBuilder::trajectories starting trajectories created "<<endl;
#endif  
  if(startingTraj.empty()) {
#ifdef DEBUG
        cout<<" Problem with starting trajectory "<<endl; 
#endif
  return result;
  }

  /// limitedCandidates( startingTraj, regionalCondition, result);
  /// FIXME: restore regionalCondition

  limitedCandidates( startingTraj, result);
#ifdef DEBUG  
   cout<<" HICTrajectoryBuilder::trajectories candidates found "<<result.size()<<endl;
#endif

  return result;
}

TempTrajectory HICTrajectoryBuilder::
createStartingTrajectory( const TrajectorySeed& seed) const
{
#ifdef DEBUG
  cout<<" HICTrajectoryBuilder::createStartingTrajectory "<<seed.direction()<<endl;
#endif  
  TempTrajectory result( seed, oppositeToMomentum );
  theForwardPropagator = &(*thePropagatorOpposite);
  theBackwardPropagator = &(*thePropagatorAlong);

  std::vector<TM> seedMeas = seedMeasurements(seed);

#ifdef DEBUG  
  std::cout<<" Size of seed "<<seedMeas.size()<<endl;
#endif  
  if ( !seedMeas.empty()) {
#ifdef DEBUG
  std::cout<<" TempTrajectory "<<std::endl;
#endif
    for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
    
       result.push(*i); 
      
                 
    }
  }
   
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
#ifdef DEBUG
  cout<<" HICTrajectoryBuilder::limitedCandidates "<<candidates.size()<<endl;
#endif   

  int theIniSign = 1;
  dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->setSign(theIniSign);
#ifdef DEBUG
  cout<<" Number of measurements "<<startingTraj.measurements().size()<<endl;
#endif
//
// Calculate the number of final trajectories. Allow no more then 4.
//
   
  int numtraj = 0; 
   
  while ( !candidates.empty()) {
#ifdef DEBUG
  cout<<" HICTrajectoryBuilder::limitedCandidates::cycle "<<candidates.size()<<endl;
#endif
    newCand.clear();

    for (TempTrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
#ifdef DEBUG	 
	 cout<< " Before findCompatibleMeasurements "<<endl;
#endif
      std::vector<TM> meas = findCompatibleMeasurements(*traj);
#ifdef DEBUG
	 cout<< " After findCompatibleMeasurements "<<meas.size()<<endl;
#endif

      if ( meas.empty()) {
#ifdef DEBUG
        cout<<": Measurements empty : "<<endl;
#endif
	if ( qualityFilter( *traj)) {addToResult( *traj, result); numtraj++;}
      }
      else {
#ifdef DEBUG
        cout<<" : Update trajectoris :   "<<endl;
#endif
	for( std::vector<TM>::const_iterator itm = meas.begin(); 
                                       	     itm != meas.end(); itm++) {
	  TempTrajectory newTraj = *traj;
	  bool good = updateTrajectory( newTraj, *itm);
          if(good)
          {
	    if ( toBeContinued(newTraj)) {
	                newCand.push_back(newTraj);
#ifdef DEBUG
               cout<<": toBeContinued : increase "<<newCand.size()<<" maximal size "<<theMaxCand<<endl;
#endif
	     }
	     else {
#ifdef DEBUG
               cout<<": good TM : to be stored :"<<endl;
#endif
	       if ( qualityFilter(newTraj)) {addToResult( newTraj, result); numtraj++;}
	    //// don't know yet
	          }
          } // good
            else
            {
#ifdef DEBUG
                  cout<<": bad TM : to be stored :"<<endl;
#endif
               if ( qualityFilter( *traj)) {addToResult( *traj, result);numtraj++;}
            }
	 } //meas 
      }
    
//      if ((int)newCand.size() > theMaxCand) {
//	sort( newCand.begin(), newCand.end(), TrajCandLess<TempTrajectory>(theLostHitPenalty));
//	newCand.erase( newCand.begin()+theMaxCand, newCand.end());
//      }

        if(numtraj > 4) break;
    } // trajectories
        if(numtraj > 4) break;
        candidates.swap(newCand);
  }

#ifdef DEBUG 
  std::cout<<" qualityFilter::Number of trajectories "<<numtraj<<std::endl;
#endif
  
}



#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoHIMuon/HiMuSeed/interface/DiMuonTrajectorySeed.h"

std::vector<TrajectoryMeasurement> 
HICTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
//  TrajectoryStateTransform tsTransform;

#ifdef DEBUG 
  cout<<" HICTrajectoryBuilder::seedMeasurements number of TM "<<dynamic_cast<DiMuonTrajectorySeed*>(const_cast<TrajectorySeed*>(&seed))->measurements().size()<<endl;
#endif  

   std::vector<TrajectoryMeasurement> start = dynamic_cast<DiMuonTrajectorySeed*>(const_cast<TrajectorySeed*>(&seed))->measurements();
   for(std::vector<TrajectoryMeasurement>::iterator imh = start.begin(); imh != start.end(); imh++)
   { 
#ifdef DEBUG       
     cout<<" HICTrajectoryBuilder::seedMeasurements::RecHit "<<endl;
#endif       
     result.push_back(*imh);
   }

  return result;
}

 bool HICTrajectoryBuilder::qualityFilter( const TempTrajectory& traj) const
{
#ifdef DEBUG
    cout << "qualityFilter called for trajectory with " 
         << traj.foundHits() << " found hits and Chi2 = "
         << traj.chiSquared() << endl;
#endif
  if ( traj.foundHits() < theMinimumNumberOfHits) {
    return false;
  }

  Trajectory traj0 = traj.toTrajectory();
// Check the number of pixels
  const Trajectory::DataContainer tms = traj0.measurements();
  int ipix = 0; 
  for( Trajectory::DataContainer::const_iterator itm = tms.begin(); itm != tms.end(); itm++) {
     if((*itm).layer()->subDetector() == GeomDetEnumerators::PixelEndcap || (*itm).layer()->subDetector() == GeomDetEnumerators::PixelBarrel) ipix++;
  }
  
#ifdef DEBUG
 cout<<" Number of pixels "<<ipix<<endl;
#endif

 if(ipix < 1) return false;
 
//
// Refit the trajectory
//

    reco::BeamSpot::CovarianceMatrix matrix;
    matrix(2,2) = 0.001;
    matrix(3,3) = 0.001;

    reco::BeamSpot bs( reco::BeamSpot::Point(0., 0., theHICConst->zvert),
                                                     0.1,
                                                     0.,
                                                     0.,
                                                     0.,
                                                    matrix
                     );
    
    enum RefitDirection{insideOut,outsideIn,undetermined};
    
    Trajectory::ConstRecHitContainer recHitsForReFit = traj0.recHits();
    
    PTrajectoryStateOnDet garbage1;
    edm::OwnVector<TrackingRecHit> garbage2;
    TrajectoryStateOnSurface firstTSOS = traj0.firstMeasurement().updatedState(); 

    PropagationDirection propDir = oppositeToMomentum;

    TrajectorySeed seed(garbage1,garbage2,propDir);
    vector<Trajectory> trajectories = theFitterTrack->fit(seed,recHitsForReFit,firstTSOS);

    TrajectoryStateOnSurface innertsos;

    if (traj0.direction() == alongMomentum) {
   //    cout<<" Direction is along momentum "<<endl;
      innertsos = traj0.firstMeasurement().updatedState();
    } else {
      innertsos = traj0.lastMeasurement().updatedState();
   //   cout<<" Direction is opposite to momentum "<<endl;
      
    }

    TSCBLBuilderNoMaterial tscblBuilder;
    TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(innertsos.freeState()),bs);

    if (tscbl.isValid()==false) {
    //cout<<" false track "<<endl; 
     return false;
    }
    
    GlobalPoint v = tscbl.trackStateAtPCA().position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    
    if(v.perp() > 0.1 ) return false;
    if(fabs(v.z() - theHICConst->zvert ) > 0.4 ) return false;

 return true;
 
}


void HICTrajectoryBuilder::addToResult( TempTrajectory& tmptraj, 
					TrajectoryContainer& result) const
{
  Trajectory traj = tmptraj.toTrajectory();
  result.push_back( traj);
}

bool HICTrajectoryBuilder::updateTrajectory( TempTrajectory& traj,
					     const TM& tm) const
{
  bool good=false; 
#ifdef DEBUG
  std::cout<<"HICTrajectoryBuilder::updateTrajectory::start"<<std::endl;
#endif
  TSOS predictedState = tm.predictedState();
  TM::ConstRecHitPointer hit = tm.recHit();
  Trajectory traj0 = traj.toTrajectory();
  
// My update
   vector<double> theCut = dynamic_cast<HICMeasurementEstimator*>
                          (const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->setCuts(traj0,tm.layer());
   int icut = 3;
   dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->chooseCuts(icut);
   const MagneticField * mf = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->getField();
   HICMuonUpdator hicup(theCut[2],theCut[3], mf,theHICConst);
   double chi2rz,chi2rf;
 
  if ( hit->isValid()) {
// Check the charge
// if() { 

   TrajectoryMeasurement tmlast = traj.lastMeasurement();
   TM::ConstRecHitPointer lasthit = tmlast.recHit();
   double dfi1 = predictedState.freeTrajectoryState()->position().phi() - lasthit->globalPosition().phi();
   double dfi2 = hit->globalPosition().phi() - lasthit->globalPosition().phi();
   

   if(dfi1*dfi2<0) {
//    cout<<" Need to change charge "<<endl;
    TrackCharge aCharge = -1*predictedState.freeTrajectoryState()->charge();
    GlobalPoint xnew = predictedState.globalPosition();
    GlobalVector pnew = predictedState.globalMomentum();
    TrajectoryStateOnSurface tsos(
                            GlobalTrajectoryParameters(xnew, pnew, aCharge, predictedState.magneticField()),
                            predictedState.curvilinearError(), predictedState.surface());
    predictedState = tsos;
   }
   
// Update trajectory
//
  TrajectoryStateOnSurface newUpdateState=hicup.update(traj0, predictedState, tm, tm.layer(), chi2rz, chi2rf);

  bool accept=
              (dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->estimate(newUpdateState,*hit)).first;
  if(accept)
  {
#ifdef DEBUG
  std::cout<<" updateTrajectory::UpdateState::New momentum "<<newUpdateState.freeTrajectoryState()->momentum().perp()<<" "<<newUpdateState.freeTrajectoryState()->momentum().z()<<std::endl;
#endif
  TM tmp = TM(predictedState, newUpdateState, hit, tm.estimate(), tm.layer());

  traj.push(tmp );
  good=true;
  return good;
  }
    else
    {
#ifdef DEBUG
       std::cout<<" updateTrajectory::failed after update "<<accept<<std::endl;
#endif
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
 // int invalidHits = 0;
  int theLowMult = 1; 

  TSOS currentState( traj.lastMeasurement().updatedState());

  vector<const DetLayer*> nl = 
                               traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());
#ifdef DEBUG
  std::cout<<" Number of layers "<<nl.size()<<std::endl;
#endif
  
  if (nl.empty()) return result;

  int seedLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                              getDetectorCode(traj.measurements().front().layer());
#ifdef DEBUG
  std::cout<<"findCompatibleMeasurements Point 0 "<<seedLayerCode<<std::endl;
#endif
							      
  int currentLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                               getDetectorCode(traj.lastLayer()); 
#ifdef DEBUG
  std::cout<<"findCompatibleMeasurements Point 1 "<<currentLayerCode<<std::endl;
#endif
  for (vector<const DetLayer*>::iterator il = nl.begin(); 
                                         il != nl.end(); il++) {

   int nextLayerCode = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->
                                                               getDetectorCode((*il)); 
#ifdef DEBUG
  std::cout<<"findCompatibleMeasurements Point 2 "<<nextLayerCode<<std::endl;
#endif

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
#ifdef DEBUG   
  std::cout<<" findCompatibleMeasurements Point 3 "<<nextLayerCode<<std::endl;
#endif   
     								       
  Trajectory traj0 = traj.toTrajectory();
  
  vector<double> theCut = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->setCuts(traj0,(*il));
  
  // Choose Win
  int icut = 1;
  dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->chooseCuts(icut);
#ifdef DEBUG
  std::cout<<" findCompatibleMeasurements::current state : "<<
             " charge "<< currentState.freeTrajectoryState()->parameters().charge()<<
             " pt "<<currentState.freeTrajectoryState()->parameters().momentum().perp()<<
             " pz "<<currentState.freeTrajectoryState()->parameters().momentum().z()<<
             " r  "<<currentState.freeTrajectoryState()->parameters().position().perp()<<
             " phi  "<<currentState.freeTrajectoryState()->parameters().position().phi()<<
             " z  "<<currentState.freeTrajectoryState()->parameters().position().z()<<
             endl; 
#endif
  const MagneticField * mf = dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->getField();
  vector<TM> tmp0;

//
// We must check the charge of the high pt track after first step
//

  if(abs(currentLayerCode) > 100&&traj0.measurements().size()>1) {
           HICMuonPropagator hmp(mf);
#ifdef DEBUG
  std::cout<<" findCompatibleMeasurements::HICMuonPropagator::for forward::start "<<std::endl;
#endif	   
               tmp0 = 
                      theLayerMeasurements->measurements((**il), currentState, hmp, *theEstimator); 
      for( vector<TM>::iterator itm = tmp0.begin(); itm != tmp0.end(); itm++ )
     {
        TM tm = (*itm);
        TSOS predictedState = tm.predictedState();
	TM::ConstRecHitPointer  hit = tm.recHit();
	TSOS updateState = traj0.lastMeasurement().updatedState();

	std::cout<<" findCompatibleMeasurements::Size of trajectory "<<traj0.measurements().size()<<
                   " valid updated state "<< updateState.isValid()<<" Predicted state is valid "
                    <<predictedState.isValid()<<
                   " charge "<< predictedState.freeTrajectoryState()->parameters().charge()<<
                   " pt "<<predictedState.freeTrajectoryState()->parameters().momentum().perp()<<
                   " pz "<<predictedState.freeTrajectoryState()->parameters().momentum().z()<<
                   " r  "<<predictedState.freeTrajectoryState()->parameters().position().perp()<<
                   " phi  "<<predictedState.freeTrajectoryState()->parameters().position().phi()<<
                   " z  "<<predictedState.freeTrajectoryState()->parameters().position().z()<<
	 std::endl;        
     }
#ifdef DEBUG
  std::cout<<" findCompatibleMeasurements::HICMuonPropagator::for forward::end "<<std::endl;
#endif             
  }




//  else
//  {
               tmp0 = 
                      theLayerMeasurements->measurements((**il), currentState, *theForwardPropagator, *theEstimator); 
//  }		      
		      
#ifdef DEBUG 
  std::cout<<" findCompatibleMeasurements Point 6 "<<theCut[0]<<" "<<theCut[1]<<std::endl;
  std::cout<<" findCompatibleMeasurements Point 7 "<<traj0.measurements().size()<<std::endl;
#endif
//   
// ========================= Choose Cut and filter =================================
//
     vector<TM> tmp;
     icut = 2;
     dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->chooseCuts(icut);
#ifdef DEBUG
     std::cout<<" findCompatibleMeasurements Point 8 "<<theCut[0]<<" "<<theCut[1]<<" Size of candidates "<<tmp0.size()<<std::endl;
#endif  
    int numtmp = 0; 
     for( vector<TM>::iterator itm = tmp0.begin(); itm != tmp0.end(); itm++ )
     {
        TM tm = (*itm);
        TSOS predictedState = tm.predictedState();
	TM::ConstRecHitPointer  hit = tm.recHit();
	TSOS updateState = traj0.lastMeasurement().updatedState();
//
// If track is not valid - stop with this hit
//
        if(!(*hit).isValid()) {
#ifdef DEBUG
         cout<<" findCompatibleMeasurements::hit is not valid "<<endl;
#endif
         continue;
        } 

#ifdef DEBUG
	std::cout<<" findCompatibleMeasurements::Size of trajectory "<<traj0.measurements().size()
                 <<" Number of TM "<< numtmp<<
                   " valid updated state "<< updateState.isValid()<<" Predicted state is valid "
                    <<predictedState.isValid()<<
                   " charge "<< predictedState.freeTrajectoryState()->parameters().charge()<<
                   " pt "<<predictedState.freeTrajectoryState()->parameters().momentum().perp()<<
                   " pz "<<predictedState.freeTrajectoryState()->parameters().momentum().z()<<
                   " r  "<<predictedState.freeTrajectoryState()->parameters().position().perp()<<
                   " phi  "<<predictedState.freeTrajectoryState()->parameters().position().phi()<<
                   " z  "<<predictedState.freeTrajectoryState()->parameters().position().z()<<
	 std::endl;
#endif

  if( traj0.measurements().size() == 1 )
  {
#ifdef DEBUG
       std::cout<<" findCompatibleMeasurements::start corrector "<<std::endl;
#endif
        HICTrajectoryCorrector theCorrector(mf,theHICConst);
#ifdef DEBUG	
	std::cout<<" findCompatibleMeasurements::corrector::initialized "<<std::endl;
#endif	
        TSOS predictedState0 = theCorrector.correct( (*traj0.lastMeasurement().updatedState().freeTrajectoryState()), 
                                                      (*(predictedState.freeTrajectoryState())), 
			                              hit->det() );
#ifdef DEBUG						      
     std::cout<<" findCompatibleMeasurements::end corrector "<<std::endl; 
#endif
      if(predictedState0.isValid()) 
      {
#ifdef DEBUG
              std::cout<<" Accept the corrected state "<<numtmp<<" Hit Valid "<<(*hit).isValid()<<std::endl; 
#endif
              predictedState = predictedState0;
	      
	if((*hit).isValid())
	{
#ifdef DEBUG
                 std::cout<<" findCompatibleMeasurements::end corrector::hit valid "<<std::endl;
#endif  
              bool accept= true;
              accept = (dynamic_cast<HICMeasurementEstimator*>(const_cast<Chi2MeasurementEstimatorBase*>(theEstimator))->estimate(predictedState,*hit)).first; 
	      
              if(!accept) {
#ifdef DEBUG
	          std::cout<<" findCompatibleMeasurements::failed after the first step "<<numtmp<<std::endl;
#endif
                  numtmp++;
	          continue;
	      } // accept
#ifdef DEBUG
             std::cout<<" findCompatibleMeasurements::estimate at the first step is ok "<<numtmp<<std::endl;
#endif
// Add trajectory measurements
             tmp.push_back(TM(predictedState, updateState, hit, tm.estimate(), tm.layer())); 
#ifdef DEBUG
             std::cout<<" findCompatibleMeasurements::fill estimate "<<numtmp<<std::endl;
#endif
        } // Hit Valid
	
       } // predicted state is valid
   } // first step
      else
      { 
//          tmp.push_back(TM(predictedState, updateState, hit, tm.estimate(), tm.layer()));
#ifdef DEBUG
            std::cout<<" findCompatibleMeasurements::Add TM to collection::Predicted state is valid "<<predictedState.isValid()<<" Hit is valid "<<(*hit).isValid()<<std::endl;
#endif	    
            if( predictedState.isValid() && (*hit).isValid() ) tmp.push_back(*itm);
      } 
          numtmp++;

  }		// tm 			   
#ifdef DEBUG
        std::cout<<" findCompatibleMeasurements::point 9 "<<std::endl;
#endif
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
#ifdef DEBUG
       std::cout<<" findCompatibleMeasurements::point 10 "<<std::endl;
#endif
  return result;
}

