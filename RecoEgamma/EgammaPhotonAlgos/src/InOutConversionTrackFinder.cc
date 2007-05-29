#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
//
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
//
#include "DataFormats/Common/interface/OwnVector.h"
//
#include "Utilities/General/interface/precomputed_value_sort.h"

#include <sstream>


InOutConversionTrackFinder::InOutConversionTrackFinder(const edm::EventSetup& es, const edm::ParameterSet& conf, const MagneticField* field,  const MeasurementTracker* theInputMeasurementTracker ) :  ConversionTrackFinder( field, theInputMeasurementTracker) , conf_(conf) {

  seedClean_ = conf_.getParameter<bool>("inOutSeedCleaning");
  smootherChiSquare_ = conf_.getParameter<double>("smootherChiSquareCut");   

  edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;
  theInitialState_       = new TransientInitialStateEstimator( es,  tise_params);

 
  std::string trajectoryBuilderName = conf_.getParameter<std::string>("TrajectoryBuilder");
  edm::ESHandle<TrackerTrajectoryBuilder> theTrajectoryBuilderHandle;
  es.get<CkfComponentsRecord>().get(trajectoryBuilderName,theTrajectoryBuilderHandle);
  theCkfTrajectoryBuilder_ = theTrajectoryBuilderHandle.product();

  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  trackerGeom= trackerHandle.product();



  theTrajectoryCleaner_ = new TrajectoryCleanerBySharedHits();


}


InOutConversionTrackFinder::~InOutConversionTrackFinder() {

  delete theTrajectoryCleaner_;
  delete theInitialState_;
}




std::vector<Trajectory> InOutConversionTrackFinder::tracks(const TrajectorySeedCollection  inOutSeeds, 
                                                           TrackCandidateCollection &output_p ) const {



  LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder::tracks getting " <<  inOutSeeds.size() << " In-Out seeds " << "\n"; 
   
  std::vector<Trajectory> tmpO;
  tmpO.erase(tmpO.begin(), tmpO.end() ) ;
  
  std::vector<Trajectory> result;
  result.erase(result.begin(), result.end() ) ;


  std::vector<Trajectory> rawResult;
  rawResult.erase(rawResult.begin(), rawResult.end() ) ;



  // Loop over the seeds
  for(TrajectorySeedCollection::const_iterator iSeed=inOutSeeds.begin(); iSeed!=inOutSeeds.end();iSeed++){

    
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
		
      }
    }

  }
   
 
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
  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
    LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  tmpO num of hits " << (*it).foundHits() << " before ordering " << "\n"; 
    
  }
  
  precomputed_value_sort( tmpO.begin(), tmpO.end(), ExtractNumOfHits()  ); 
  
  
  LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  tmpO after sorting " << "\n"; 
  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
    LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  tmpO  num of hits " << (*it).foundHits() << "\n"; 


  }

  for (int i=tmpO.size()-1; i>=0; i--) {
    unsmoothedResult.push_back(  tmpO[i] );  
  }
  LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  unsmoothedResult size  " <<  unsmoothedResult.size() << "\n";   

  for (std::vector<Trajectory>::const_iterator it =  unsmoothedResult.begin(); it !=  unsmoothedResult.end(); it++) {
    LogDebug("InOutConversionTrackFinder") << " InOutConversionTrackFinder  unsmoothedResult  after reordering " <<(*it).foundHits() <<  "\n"; 

  }


  // Convert to TrackCandidates and fill in the output_p
  for (std::vector<Trajectory>::const_iterator it = unsmoothedResult.begin(); it != unsmoothedResult.end(); it++) {
    
    edm::OwnVector<TrackingRecHit> recHits;
    Trajectory::RecHitContainer thits = it->recHits();
    for (Trajectory::RecHitContainer::const_iterator hitIt = thits.begin(); hitIt != thits.end(); hitIt++) {
      recHits.push_back( (**hitIt).hit()->clone());
    }
    
    
    std::pair<TrajectoryStateOnSurface, const GeomDet*> initState =  theInitialState_->innerState( *it);
    
    // temporary protection againt invalid initial states
    if (! initState.first.isValid() || initState.second == 0) {
      //cout << "invalid innerState, will not make TrackCandidate" << endl;
      continue;
    }
    
    PTrajectoryStateOnDet* state = TrajectoryStateTransform().persistentState( initState.first, initState.second->geographicalId().rawId());

    LogDebug("InOutConversionTrackFinder") << "  InOutConversionTrackFinder::track Making the result: seed position " << it->seed().startingState().parameters().position()  << " seed momentum " <<  it->seed().startingState().parameters().momentum() << " charge " <<  it->seed().startingState().parameters().charge () << "\n";
    LogDebug("InOutConversionTrackFinder") << "  InOutConversionTrackFinder::track TSOS charge  " << initState.first.charge() << "\n";
    
    LogDebug("InOutConversionTrackFinder") <<   " InOutConversionTrackFinder::track  PTrajectoryStateOnDet* state position  " << state->parameters().position() << " momentum " << state->parameters().momentum() << " charge " <<   state->parameters().charge () << "\n";
    

    result.push_back(*it);  
    
    output_p.push_back(TrackCandidate(recHits, it->seed(),*state ) );
    delete state;
  }
  
  LogDebug("InOutConversionTrackFinder") << "  InOutConversionTrackFinder::track Returning " << result.size() << " valid In Out Trajectories " << "\n";
  return  result;
}
