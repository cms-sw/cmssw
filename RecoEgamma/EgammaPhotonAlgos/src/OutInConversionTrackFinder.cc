#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
//
#include "RecoTracker/CkfPattern/interface/SeedCleanerByHitPosition.h"
#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerByHitPosition.h"
#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerBySharedInput.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
//
#include "DataFormats/Common/interface/OwnVector.h"
//
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "Utilities/General/interface/precomputed_value_sort.h"



OutInConversionTrackFinder::OutInConversionTrackFinder(const edm::EventSetup& es, 
						       const edm::ParameterSet& conf ) :  ConversionTrackFinder( es, conf )
{


  theTrajectoryCleaner_ = new TrajectoryCleanerBySharedHits(conf);

  // get the seed cleaner
  std::string cleaner = conf_.getParameter<std::string>("OutInRedundantSeedCleaner");
  if (cleaner == "SeedCleanerByHitPosition") {
    theSeedCleaner_ = new SeedCleanerByHitPosition();
  } else if (cleaner == "CachingSeedCleanerByHitPosition") {
    theSeedCleaner_ = new CachingSeedCleanerByHitPosition();
  } else if (cleaner == "CachingSeedCleanerBySharedInput") {
    theSeedCleaner_ = new CachingSeedCleanerBySharedInput();
  } else if (cleaner == "none") {
    theSeedCleaner_ = 0;
  } else {
    throw cms::Exception("OutInRedundantSeedCleaner not found", cleaner);
  }
  

 
 
  
}




OutInConversionTrackFinder::~OutInConversionTrackFinder() {

  delete theTrajectoryCleaner_;
  if (theSeedCleaner_) delete theSeedCleaner_;

}


std::vector<Trajectory> OutInConversionTrackFinder::tracks(const TrajectorySeedCollection& outInSeeds, 
							   TrackCandidateCollection &output_p ) const { 

  
  //  std::cout  << "OutInConversionTrackFinder::tracks getting " <<  outInSeeds.size() << " Out-In seeds " << "\n";;
 
  std::vector<Trajectory> tmpO;
  tmpO.erase(tmpO.begin(), tmpO.end() ) ;

  std::vector<Trajectory> result;
  result.erase(result.begin(), result.end() ) ;
  

  std::vector<Trajectory> rawResult;
  if (theSeedCleaner_) theSeedCleaner_->init( &rawResult );


  ///// This loop is only for debugging
  /*
  for(TrajectorySeedCollection::const_iterator iSeed=outInSeeds.begin(); iSeed!=outInSeeds.end();iSeed++){
    DetId tmpId = DetId( iSeed->startingState().detId());
    const GeomDet* tmpDet  = theMeasurementTracker_->geomTracker()->idToDet( tmpId );
    GlobalVector gv = tmpDet->surface().toGlobal( iSeed->startingState().parameters().momentum() );
    
    LogDebug("OutInConversionTrackFinder") << " OutInConversionTrackFinder::tracks hits in the seed " << iSeed->nHits() << "\n";
    LogDebug("OutInConversionTrackFinder")<< " OutInConversionTrackFinder::tracks seed starting state position  " << iSeed->startingState().parameters().position() << " momentum " <<  iSeed->startingState().parameters().momentum() << " charge " << iSeed->startingState().parameters().charge() << " R " << gv.perp() << " eta " << gv.eta() << " phi " << gv.phi() << "\n";
    
    TrajectorySeed::range hitRange = iSeed->recHits();
    for (TrajectorySeed::const_iterator ihit = hitRange.first; ihit != hitRange.second; ihit++) {
      
      if ( ihit->isValid() ) {
	
	LogDebug("OutInConversionTrackFinder")  << " Valid hit global position " << theMeasurementTracker_->geomTracker()->idToDet((ihit)->geographicalId())->surface().toGlobal((ihit)->localPosition()) << " R " << theMeasurementTracker_->geomTracker()->idToDet((ihit)->geographicalId())->surface().toGlobal((ihit)->localPosition()).perp() << " phi " << theMeasurementTracker_->geomTracker()->idToDet((ihit)->geographicalId())->surface().toGlobal((ihit)->localPosition()).phi() << " eta " << theMeasurementTracker_->geomTracker()->idToDet((ihit)->geographicalId())->surface().toGlobal((ihit)->localPosition()).eta() <<    "\n" ;
	
      }
    }
  } 
  
  */


  
  
  
  int goodSeed=0;  
  std::vector<Trajectory> theTmpTrajectories;
  for(TrajectorySeedCollection::const_iterator iSeed=outInSeeds.begin(); iSeed!=outInSeeds.end();iSeed++){

    theTmpTrajectories.clear();
    
    if (!theSeedCleaner_ || theSeedCleaner_->good(&(*iSeed))) {
      goodSeed++;


      DetId tmpId = DetId( iSeed->startingState().detId());
      const GeomDet* tmpDet  = theMeasurementTracker_->geomTracker()->idToDet( tmpId );
      GlobalVector gv = tmpDet->surface().toGlobal( iSeed->startingState().parameters().momentum() );
      
      //      std::cout << " OutInConversionTrackFinder::tracks hits in the seed " << iSeed->nHits() << "\n";
      LogDebug("OutInConversionTrackFinder") << " OutInConversionTrackFinder::tracks seed starting state position  " << iSeed->startingState().parameters().position() << " momentum " <<  iSeed->startingState().parameters().momentum() << " charge " << iSeed->startingState().parameters().charge() << " R " << gv.perp() << " eta " << gv.eta() << " phi " << gv.phi() << "\n";
      
      
      theCkfTrajectoryBuilder_->trajectories(*iSeed, theTmpTrajectories);
      
      LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder::track returned " << theTmpTrajectories.size() << " trajectories" << "\n";
      
      theTrajectoryCleaner_->clean(theTmpTrajectories);
      
      for(std::vector<Trajectory>::const_iterator it=theTmpTrajectories.begin();
	  it!=theTmpTrajectories.end(); it++){
	if( it->isValid() ) {
	  rawResult.push_back(*it);
	  if (theSeedCleaner_) theSeedCleaner_->add( & (*it) );
	}
      }
  
      
    }
  }  // end loop over the seeds 
  LogDebug("OutInConversionTrackFinder") << " OutInConversionTrackFinder::track Good seeds " << goodSeed   << "\n";
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder::track rawResult size after cleaning " << rawResult.size() << "\n";

  if (theSeedCleaner_) theSeedCleaner_->done();
  
  std::vector<Trajectory> unsmoothedResult;
  theTrajectoryCleaner_->clean(rawResult);
  
  for (std::vector<Trajectory>::const_iterator itraw = rawResult.begin(); itraw != rawResult.end(); itraw++) {
    if((*itraw).isValid()) {
      tmpO.push_back( *itraw );
      LogDebug("OutInConversionTrackFinder") << " rawResult num hits " << (*itraw).foundHits() << "\n";
    }
  }
  
  
  
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO size " << tmpO.size() << " before sorting " << "\n"; 
  //  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
  // LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO num of hits " << (*it).foundHits() << " before ordering " << "\n"; 
  // }
  
  precomputed_value_sort( tmpO.begin(), tmpO.end(), ExtractNumOfHits()  ); 

  
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO after sorting " << "\n"; 
  //  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
  // LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO  num of hits " << (*it).foundHits() << "\n"; 
  // }
  
  for (int i=tmpO.size()-1; i>=0; i--) {
    unsmoothedResult.push_back(  tmpO[i] );  
  }
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  unsmoothedResult size  " <<  unsmoothedResult.size() << "\n";   
  
  // for (std::vector<Trajectory>::const_iterator it =  unsmoothedResult.begin(); it !=  unsmoothedResult.end(); it++) {
  // LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  unsmoothedResult  after reordering " <<(*it).foundHits() <<  "\n"; 
  //  }


  
  // Check if the inner state is valid
  tmpO.clear();
  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO size " << tmpO.size() << " after clearing " << "\n"; 
  for (std::vector<Trajectory>::const_iterator it =  unsmoothedResult.begin(); it != unsmoothedResult.end(); it++) {
    if( !it->isValid() ) continue;

    std::pair<TrajectoryStateOnSurface, const GeomDet*> initState =  theInitialState_->innerState( *it);
    //  LogDebug("OutInConversionTrackFinder") << " Initial state parameters " << initState.first << "\n";    
    
    // temporary protection againt invalid initial states
    if (! initState.first.isValid() || initState.second == 0) {
      LogDebug("OutInConversionTrackFinder")  << "invalid innerState, will not make TrackCandidate" << "\n";;
      continue;
    }
    tmpO.push_back(*it);
  }


  LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  tmpO size " << tmpO.size() << " after filling " << "\n"; 
  if ( tmpO.size() ) {
    std::vector<Trajectory>::iterator it=tmpO.begin();
    
    // only send out the two best tracks 
    result.push_back(*it);      
    if ( tmpO.size() > 1) result.push_back(*(++it));
  }
  
  //  for (std::vector<Trajectory>::const_iterator it =result.begin(); it != result.end(); it++) {
  // LogDebug("OutInConversionTrackFinder") << "OutInConversionTrackFinder  Result  num of hits " << (*it).foundHits() << "\n"; 
  //}
  
  
  // Converted to track candidates  
  for (std::vector<Trajectory>::const_iterator it =  result.begin(); it != result.end(); it++) {
    //    if( !it->isValid() ) continue;

    edm::OwnVector<TrackingRecHit> recHits;
    Trajectory::RecHitContainer thits;
    it->recHitsV(thits,useSplitHits_);
    recHits.reserve(thits.size());
    for (Trajectory::RecHitContainer::const_iterator hitIt = thits.begin(); hitIt != thits.end(); hitIt++) {
      recHits.push_back( (**hitIt).hit()->clone());
    }
    
    
    std::pair<TrajectoryStateOnSurface, const GeomDet*> initState =  theInitialState_->innerState( *it);
    // temporary protection againt invalid initial states
    if (! initState.first.isValid() || initState.second == 0) {
      //cout << "invalid innerState, will not make TrackCandidate" << endl;
      continue;
    }

    PTrajectoryStateOnDet state;
    if(useSplitHits_ && (initState.second != thits.front()->det()) && thits.front()->det() ){ 
      TrajectoryStateOnSurface propagated = thePropagator_->propagate(initState.first,thits.front()->det()->surface());
      if (!propagated.isValid()) continue;
      state = trajectoryStateTransform::persistentState(propagated,
							 thits.front()->det()->geographicalId().rawId());
    }
    else  state = trajectoryStateTransform::persistentState( initState.first,
								   initState.second->geographicalId().rawId());

    LogDebug("OutInConversionTrackFinder")<< "OutInConversionTrackFinder  Number of hits for the track candidate " << recHits.size() << " TSOS charge " << initState.first.charge() << "\n";  
    output_p.push_back(TrackCandidate(recHits, it->seed(),state ) );
  }  
  
  
  //  std::cout << "  Returning " << result.size() << "Out In Trajectories  " << "\n";      
  

  return  result;
  


}
