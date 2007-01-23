#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/CkfPattern/interface/GroupedTrajCandLess.h"
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
//
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
//
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "Utilities/General/interface/precomputed_value_sort.h"



OutInConversionTrackFinder::OutInConversionTrackFinder(const edm::EventSetup& es, const edm::ParameterSet& conf, const MagneticField* field,  const MeasurementTracker* theInputMeasurementTracker ) :  ConversionTrackFinder(  field, theInputMeasurementTracker), conf_(conf)
{


  
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder CTOR  theMeasurementTracker_   " << theMeasurementTracker_<<  "\n"; 
  
  

  
  seedClean_ = conf_.getParameter<bool>("outInSeedCleaning");
  // get nested parameter set for the TransientInitialStateEstimator
  edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;
  theInitialState_       = new TransientInitialStateEstimator( es,  tise_params );

  // Get the TrajectoryBuilder  
  std::string trajectoryBuilderName = conf_.getParameter<std::string>("TrajectoryBuilder");
  edm::ESHandle<TrackerTrajectoryBuilder> theTrajectoryBuilderHandle;
  es.get<CkfComponentsRecord>().get(trajectoryBuilderName,theTrajectoryBuilderHandle);
  theCkfTrajectoryBuilder_ = theTrajectoryBuilderHandle.product();
  //
  theTrajectoryCleaner_ = new TrajectoryCleanerBySharedHits();
  

}


OutInConversionTrackFinder::~OutInConversionTrackFinder() {

  // delete theCkfTrajectoryBuilder_;
  delete theTrajectoryCleaner_;
  delete  theInitialState_;

}



// std::auto_ptr<TrackCandidateCollection>  OutInConversionTrackFinder::tracks(const TrajectorySeedCollection outInSeeds  )const  {

//std::vector<Trajectory> OutInConversionTrackFinder::tracks(const TrajectorySeedCollection outInSeeds, 
//                                                                TrackCandidateCollection &output_p,   
//                                                                reco::TrackCandidateSuperClusterAssociationCollection& outAssoc, int iSC )const  {

std::vector<Trajectory> OutInConversionTrackFinder::tracks(const TrajectorySeedCollection outInSeeds, 
							   TrackCandidateCollection &output_p ) const { 

  
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder::tracks getting " <<  outInSeeds.size() << " Out-In seeds " << endl;



  std::vector<Trajectory> tmpO;
  tmpO.erase(tmpO.begin(), tmpO.end() ) ;

  std::vector<Trajectory> result;
  result.erase(result.begin(), result.end() ) ;


  std::vector<Trajectory> rawResult;
  rawResult.erase(rawResult.begin(), rawResult.end() ) ;


  for(TrajectorySeedCollection::const_iterator iSeed=outInSeeds.begin(); iSeed!=outInSeeds.end();iSeed++){

    /*    
    LogDebug("OutInConversionTrackFinder") << " OutInConversionTrackFinder::tracks hits in the seed " << iSeed->nHits() << "\n";
    LogDebug("OutInConversionTrackFinder") << " OutInConversionTrackFinder::tracks seed starting state position  " << iSeed->startingState().parameters().position() << " momentum " <<  iSeed->startingState().parameters().momentum() << " charge " << iSeed->startingState().parameters().charge() << "\n";
    LogDebug("OutInConversionTrackFinder") << " OutInConversionTrackFinder::tracks seed  starting state para, vector  " << iSeed->startingState().parameters().vector() << "\n";
    
    */
    
    std::vector<Trajectory> theTmpTrajectories;

    theTmpTrajectories = theCkfTrajectoryBuilder_->trajectories(*iSeed);
    
    std::cout << "OutInConversionTrackFinder::track returned " << theTmpTrajectories.size() << " trajectories" << "\n";
    
    theTrajectoryCleaner_->clean(theTmpTrajectories);
    
    for(std::vector<Trajectory>::const_iterator it=theTmpTrajectories.begin();
	it!=theTmpTrajectories.end(); it++){
      if( it->isValid() ) {
	rawResult.push_back(*it);
      }
    }
    LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder::track rawResult size after cleaning " << rawResult.size() << "\n";

  }
  
  
  
  std::vector<Trajectory> unsmoothedResult;
  theTrajectoryCleaner_->clean(rawResult);
  
  for (std::vector<Trajectory>::const_iterator itraw = rawResult.begin(); itraw != rawResult.end(); itraw++) {
    if((*itraw).isValid()) {
      //      unsmoothedResult.push_back( *itraw);
      tmpO.push_back( *itraw );
      LogDebug("OutInConversionTrackFinder") << " rawResult num hits " << (*itraw).foundHits() << "\n";
    }
  }
  
    
  
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO size " << tmpO.size() << " before sorting " << "\n"; 
  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
    LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO num of hits " << (*it).foundHits() << " before ordering " << "\n"; 
    
  }
  
  precomputed_value_sort( tmpO.begin(), tmpO.end(), ExtractNumOfHits()  ); 


  

  
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO after sorting " << "\n"; 
  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
    LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO  num of hits " << (*it).foundHits() << "\n"; 


  }

  for (int i=tmpO.size()-1; i>=0; i--) {
    unsmoothedResult.push_back(  tmpO[i] );  
  }
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  unsmoothedResult size  " <<  unsmoothedResult.size() << "\n";   

  for (std::vector<Trajectory>::const_iterator it =  unsmoothedResult.begin(); it !=  unsmoothedResult.end(); it++) {
    LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  unsmoothedResult  after reordering " <<(*it).foundHits() <<  "\n"; 

  }



 
  if ( unsmoothedResult.size() ) {
    vector<Trajectory>::iterator it=unsmoothedResult.begin();

    // only send out the two best tracks 
    result.push_back(*it);      
    if ( unsmoothedResult.size() > 1) result.push_back(*(++it));
  }

  for (std::vector<Trajectory>::const_iterator it =result.begin(); it != result.end(); it++) {
    LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  Result  num of hits " << (*it).foundHits() << "\n"; 

  }


  //LogDebug("OutInConversionTrackFinder") << "  Returning " << result.size() << "Out In Tracks " << "\n";
  LogDebug("OutInConversionTrackFinder") << "  Returning " << unsmoothedResult.size() << "Out In Trajectories  " << "\n";
 


  // Convert to TrackCandidates and fill in the output_p
  for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin(); it != unsmoothedResult.end(); it++) {
    
    edm::OwnVector<TrackingRecHit> recHits;
    Trajectory::RecHitContainer thits = it->recHits();
    for (Trajectory::RecHitContainer::const_iterator hitIt = thits.begin(); hitIt != thits.end(); hitIt++) {
      recHits.push_back( (**hitIt).hit()->clone());
    }
	
    LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  Number of hits for the track candidate " << recHits.size() << "\n";

    std::pair<TrajectoryStateOnSurface, const GeomDet*> initState =  theInitialState_->innerState( *it);
    //  LogDebug("OutInConversionTrackFinder") << " Initial state parameters " << initState.first << "\n";    

    // temporary protection againt invalid initial states
    if (! initState.first.isValid() || initState.second == 0) {
      cout << "invalid innerState, will not make TrackCandidate" << endl;
      continue;
    }
    


    PTrajectoryStateOnDet* state = TrajectoryStateTransform().persistentState( initState.first, initState.second->geographicalId().rawId());
    
    output_p.push_back(TrackCandidate(recHits, it->seed(),*state ) );
    delete state;
  }
  
      
      




  
  //  return  unsmoothedResult;
  return  result;
  


}
