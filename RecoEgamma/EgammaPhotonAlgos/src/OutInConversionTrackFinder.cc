#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
//
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
//
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
//
#include "DataFormats/Common/interface/OwnVector.h"
#include "Utilities/General/interface/precomputed_value_sort.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


OutInConversionTrackFinder::OutInConversionTrackFinder(const edm::EventSetup& es, const edm::ParameterSet& conf, const MagneticField* field,  const MeasurementTracker* theInputMeasurementTracker ) :  ConversionTrackFinder(  field, theInputMeasurementTracker), conf_(conf)
{
  std::cout << " OutInConversionTrackFinder CTOR  theMeasurementTracker_   " << theMeasurementTracker_<<  std::endl; 
  
  
  seedClean_ = conf_.getParameter<bool>("outInSeedCleaning");
  // get nested parameter set for the TransientInitialStateEstimator

  edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;

  theInitialState_       = new TransientInitialStateEstimator( es,  tise_params );
  
  //  theCkfTrajectoryBuilder_ = new CkfTrajectoryBuilder(conf_,es,theMeasurementTracker_);
  
  std::string trajectoryBuilderName = conf_.getParameter<std::string>("TrajectoryBuilder");
  edm::ESHandle<TrackerTrajectoryBuilder> theTrajectoryBuilderHandle;

  es.get<CkfComponentsRecord>().get(trajectoryBuilderName,theTrajectoryBuilderHandle);
  theCkfTrajectoryBuilder_ = theTrajectoryBuilderHandle.product();

  theTrajectoryCleaner_ = new TrajectoryCleanerBySharedHits();
  

}


OutInConversionTrackFinder::~OutInConversionTrackFinder() {

  // delete theCkfTrajectoryBuilder_;
  delete theTrajectoryCleaner_;
  delete  theInitialState_;

}



std::vector<Trajectory>  OutInConversionTrackFinder::tracks(const TrajectorySeedCollection outInSeeds  )const  {

// TrackCandidateCollection  OutInConversionTrackFinder::tracks(const TrajectorySeedCollection outInSeeds  )const  {
  
  std::cout << " OutInConversionTrackFinder::tracks getting " <<  outInSeeds.size() << " Out-In seeds " << endl;


  std::vector<Trajectory> tmpO;
  tmpO.erase(tmpO.begin(), tmpO.end() ) ;

  std::vector<Trajectory> result;
  result.erase(result.begin(), result.end() ) ;


  std::vector<Trajectory> rawResult;
  rawResult.erase(rawResult.begin(), rawResult.end() ) ;


  for(TrajectorySeedCollection::const_iterator iSeed=outInSeeds.begin(); iSeed!=outInSeeds.end();iSeed++){
    
    std::cout << " OutInConversionTrackFinder::tracks hits in the seed " << iSeed->nHits() << std::endl;
    std::cout << " OutInConversionTrackFinder::tracks seed starting state position  " << iSeed->startingState().parameters().position() << " momentum " <<  iSeed->startingState().parameters().momentum() << " charge " << iSeed->startingState().parameters().charge() << std::endl;
    std::cout << " OutInConversionTrackFinder::tracks seed  starting state para, vector  " << iSeed->startingState().parameters().vector() << std::endl;
    
    
    
    std::vector<Trajectory> theTmpTrajectories;

    theTmpTrajectories = theCkfTrajectoryBuilder_->trajectories(*iSeed);
    
    std:: cout << " OutInConversionTrackFinder::track returned " << theTmpTrajectories.size() << " trajectories" << std::endl;
    
    theTrajectoryCleaner_->clean(theTmpTrajectories);
    
    for(std::vector<Trajectory>::const_iterator it=theTmpTrajectories.begin();
	it!=theTmpTrajectories.end(); it++){
      if( it->isValid() ) {
	rawResult.push_back(*it);
      }
    }
    std::cout << " OutInConversionTrackFinder::track rawResult size after cleaning " << rawResult.size() << std::endl;

  }
  
  
  
  std::vector<Trajectory> unsmoothedResult;
  theTrajectoryCleaner_->clean(rawResult);
  
  for (std::vector<Trajectory>::const_iterator itraw = rawResult.begin(); itraw != rawResult.end(); itraw++) {
    if((*itraw).isValid()) {
          unsmoothedResult.push_back( *itraw);
	  tmpO.push_back( *itraw );
	  std::cout << " rawResult num hits " << (*itraw).foundHits() << std::endl;
    }
  }
  
    
  
  std::cout << " OutInConversionTrackFinder  tmpO size " << tmpO.size() << " before sorting " << std::endl; 
  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
    std::cout << " OutInConversionTrackFinder  tmpO num of hits " << (*it).foundHits() << " before ordering " << std::endl; 
    
  }
  
  precomputed_value_sort( tmpO.begin(), tmpO.end(), ExtractNumOfHits()  ); 
  
  
  std::cout << " OutInConversionTrackFinder  tmpO after sorting " << std::endl; 
  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
    std::cout << " OutInConversionTrackFinder  tmpO  num of hits " << (*it).foundHits() << std::endl; 

  }


  if ( tmpO.size() ) {
    vector<Trajectory>::iterator it=tmpO.begin();

    // only send out the two best tracks 
    result.push_back(*it);      
    if ( tmpO.size() > 1) result.push_back(*(++it));
  }


  std::cout << "  Returning " << tmpO.size() << " Out In Tracks " << std::endl;
  return result;

}
