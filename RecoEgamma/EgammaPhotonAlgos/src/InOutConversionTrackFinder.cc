#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"

//
#include "RecoTracker/CkfPattern/interface/SeedCleanerByHitPosition.h"
#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerByHitPosition.h"
#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerBySharedInput.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
//
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"
//
#include "DataFormats/Common/interface/OwnVector.h"
//
#include "Utilities/General/interface/precomputed_value_sort.h"

#include <sstream>


InOutConversionTrackFinder::InOutConversionTrackFinder(const edm::EventSetup& es, 
						       const edm::ParameterSet& conf ) : ConversionTrackFinder (es,  conf ) 
{ 

 
  theTrajectoryCleaner_ = new TrajectoryCleanerBySharedHits(conf);

 // get the seed cleaner
 std::string cleaner = conf_.getParameter<std::string>("InOutRedundantSeedCleaner");
 if (cleaner == "SeedCleanerByHitPosition") {
   theSeedCleaner_ = new SeedCleanerByHitPosition();
 } else if (cleaner == "CachingSeedCleanerByHitPosition") {
   theSeedCleaner_ = new CachingSeedCleanerByHitPosition();
 } else if (cleaner == "CachingSeedCleanerBySharedInput") {
   theSeedCleaner_ = new CachingSeedCleanerBySharedInput();
 } else if (cleaner == "none") {
   theSeedCleaner_ = 0;
 } else {
   throw cms::Exception("InOutRedundantSeedCleaner not found", cleaner);
 }

}


InOutConversionTrackFinder::~InOutConversionTrackFinder() {

  delete theTrajectoryCleaner_;
  if (theSeedCleaner_) delete theSeedCleaner_;
}




std::vector<Trajectory> InOutConversionTrackFinder::tracks(const TrajectorySeedCollection&  inOutSeeds, 
                                                           TrackCandidateCollection &output_p ) const {



  //  std::cout << " InOutConversionTrackFinder::tracks getting " <<  inOutSeeds.size() << " In-Out seeds " << "\n"; 
   
  std::vector<Trajectory> tmpO;
  tmpO.erase(tmpO.begin(), tmpO.end() ) ;
  
  std::vector<Trajectory> result;
  result.erase(result.begin(), result.end() ) ;


  std::vector<Trajectory> rawResult;
   if (theSeedCleaner_) theSeedCleaner_->init( &rawResult );



  // Loop over the seeds
  int goodSeed=0;
  for(TrajectorySeedCollection::const_iterator iSeed=inOutSeeds.begin(); iSeed!=inOutSeeds.end();iSeed++){
    if (!theSeedCleaner_ || theSeedCleaner_->good(&(*iSeed))) {
    goodSeed++;
    
    LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder::tracks hits in the seed " << iSeed->nHits() << "\n";
    LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder::tracks seed starting state position  " << iSeed->startingState().parameters().position() << " momentum " <<  iSeed->startingState().parameters().momentum() << " charge " << iSeed->startingState().parameters().charge() << "\n";
    LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder::tracks seed  starting state para, vector  " << iSeed->startingState().parameters().vector() << "\n";
     

    
    std::vector<Trajectory> theTmpTrajectories;

    theTmpTrajectories = theCkfTrajectoryBuilder_->trajectories(*iSeed);
    
    LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder::track returned " << theTmpTrajectories.size() << " trajectories for this seed " << "\n";
    
    theTrajectoryCleaner_->clean(theTmpTrajectories);
    
    for(std::vector<Trajectory>::const_iterator it=theTmpTrajectories.begin(); it!=theTmpTrajectories.end(); it++){
      if( it->isValid() ) {
	rawResult.push_back(*it);
	if (theSeedCleaner_) theSeedCleaner_->add( & (*it) );	
      }
    }
    }
  }  // end loop over the seeds 
   
 

  LogDebug("InOutConversionTrackFinder") << "InOutConversionTrackFinder::track Good seeds " << goodSeed   << "\n"  ;
  LogDebug("InOutConversionTrackFinder") << "InOutConversionTrackFinder::track rawResult size after cleaning " << rawResult.size() << "\n";  

  if (theSeedCleaner_) theSeedCleaner_->done();


  std::vector<Trajectory> unsmoothedResult;
  LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder::track  Start second cleaning " << "\n";
  theTrajectoryCleaner_->clean(rawResult);
  LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder::track rawResult size after cleaning " << rawResult.size() << "\n";
  


  int tra=0;
  for (std::vector<Trajectory>::const_iterator itraw = rawResult.begin(); itraw != rawResult.end(); itraw++) {
    tra++;
    LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder looping of rawResult after cleaning " << tra << "\n";
    if((*itraw).isValid()) {
      // unsmoothedResult.push_back( *itraw);
      tmpO.push_back( *itraw );
      LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder::track rawResult num of valid recHits per trajectory " << (*itraw).foundHits() << "\n";
    }

  }
  
  LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  tmpO size " << tmpO.size() << " before sorting " << "\n"; 
  //  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
  // LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  tmpO num of hits " << (*it).foundHits() << " before ordering " << "\n"; 
  //}
  
  precomputed_value_sort( tmpO.begin(), tmpO.end(), ExtractNumOfHits()  ); 
  
  
  LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  tmpO after sorting " << "\n"; 
  //  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
  // LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  tmpO  num of hits " << (*it).foundHits() << "\n"; 
  // }

  for (int i=tmpO.size()-1; i>=0; i--) {
    unsmoothedResult.push_back(  tmpO[i] );  
  }
  LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  unsmoothedResult size  " <<  unsmoothedResult.size() << "\n";   

  //  for (std::vector<Trajectory>::const_iterator it =  unsmoothedResult.begin(); it !=  unsmoothedResult.end(); it++) {
  // LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  unsmoothedResult  after reordering " <<(*it).foundHits() <<  "\n"; 
  //  }


  // Convert to TrackCandidates and fill in the output_p
  Traj2TrackHits t2t(theCkfTrajectoryBuilder_->hitBuilder(),true);
  for (std::vector<Trajectory>::const_iterator it = unsmoothedResult.begin(); it != unsmoothedResult.end(); it++) {

     edm::OwnVector<TrackingRecHit> recHits;
     if(it->direction() != alongMomentum) LogDebug("InOutConversionTrackFinder") << "InOutConv not along momentum... " << std::endl;

     t2t(*it,recHits,useSplitHits_);
    
     assert(recHits.size()==(*it).measurements().size());
    
    /*
    edm::OwnVector<TrackingRecHit> recHits;
    Trajectory::RecHitContainer thits;
    it->recHitsV(thits,useSplitHits_);
    recHits.reserve(thits.size());
    for (Trajectory::RecHitContainer::const_iterator hitIt = thits.begin(); hitIt != thits.end(); hitIt++) {
      recHits.push_back( (**hitIt).hit()->clone());
    }
    */
    
    std::pair<TrajectoryStateOnSurface, const GeomDet*> initState =  theInitialState_->innerState( *it);


    // temporary protection againt invalid initial states
    if ( (!initState.first.isValid()) | (initState.second == nullptr)) {
      LogDebug("InOutConversionTrackFinder") << "invalid innerState, will not make TrackCandidate" << std::endl;
      continue;
    }
    
    PTrajectoryStateOnDet state;
    if(useSplitHits_ && (initState.second != recHits.front().det()) && recHits.front().det() ){ 
      TrajectoryStateOnSurface propagated = thePropagator_->propagate(initState.first,recHits.front().det()->surface());
      if (!propagated.isValid()) continue;
      state = trajectoryStateTransform::persistentState(propagated,
							 recHits.front().rawId());
    }
    else state = trajectoryStateTransform::persistentState( initState.first,
								   initState.second->geographicalId().rawId());
    
    LogDebug("InOutConversionTrackFinder") << "  InOutConversionTrackFinder::track Making the result: seed position " << it->seed().startingState().parameters().position()  << " seed momentum " <<  it->seed().startingState().parameters().momentum() << " charge " <<  it->seed().startingState().parameters().charge () << "\n";
    LogDebug("InOutConversionTrackFinder") << "  InOutConversionTrackFinder::track TSOS charge  " << initState.first.charge() << "\n";
    
    LogDebug("InOutConversionTrackFinder") <<   " InOutConversionTrackFinder::track  PTrajectoryStateOnDet* state position  " 
     << state.parameters().position() << " momentum " << state.parameters().momentum() << " charge " <<   state.parameters().charge () << "\n";
    
    result.push_back(*it);
    output_p.push_back(TrackCandidate(recHits, it->seed(),state ) );
  }
  // assert(result.size()==output_p.size());
  LogDebug("InOutConversionTrackFinder") << "  InOutConversionTrackFinder::track Returning " << result.size() << " valid In Out Trajectories " << "\n";
  return  result;
}
