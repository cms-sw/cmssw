//
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"

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
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <sstream>


InOutConversionTrackFinder::InOutConversionTrackFinder(const edm::EventSetup& es, const edm::ParameterSet& conf, const MagneticField* field,  const MeasurementTracker* theInputMeasurementTracker ) :  ConversionTrackFinder( field, theInputMeasurementTracker) , conf_(conf) {
  std::cout << " InOutConversionTrackFinder CTOR " << std:: endl;  
    
  seedClean_ = conf_.getParameter<bool>("inOutSeedCleaning");
  smootherChiSquare_ = conf_.getParameter<double>("smootherChiSquareCut");   

  edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;
  theInitialState_       = new TransientInitialStateEstimator( es,  tise_params);

  //  theCkfTrajectoryBuilder_ = new CkfTrajectoryBuilder(conf_,es,theMeasurementTracker_);
 
  std::string trajectoryBuilderName = conf_.getParameter<std::string>("TrajectoryBuilder");
  edm::ESHandle<TrackerTrajectoryBuilder> theTrajectoryBuilderHandle;
  es.get<CkfComponentsRecord>().get(trajectoryBuilderName,theTrajectoryBuilderHandle);
  theCkfTrajectoryBuilder_ = theTrajectoryBuilderHandle.product();

  theTrajectoryCleaner_ = new TrajectoryCleanerBySharedHits();


}


InOutConversionTrackFinder::~InOutConversionTrackFinder() {

  //  delete theCkfTrajectoryBuilder_;
  delete theTrajectoryCleaner_;
  delete theInitialState_;
}




std::vector<Trajectory>  InOutConversionTrackFinder::tracks(const TrajectorySeedCollection inOutSeeds )const  {
// TrackCandidateCollection InOutConversionTrackFinder::tracks(const TrajectorySeedCollection seeds )const  {
  std::cout << " InOutConversionTrackFinder::tracks getting " <<  inOutSeeds.size() << " In-Out seeds " << endl;
  
  std::vector<Trajectory> tmpO;
  tmpO.erase(tmpO.begin(), tmpO.end() ) ;
  
  std::vector<Trajectory> result;
  result.erase(result.begin(), result.end() ) ;


  std::vector<Trajectory> rawResult;
  rawResult.erase(rawResult.begin(), rawResult.end() ) ;



  for(TrajectorySeedCollection::const_iterator iSeed=inOutSeeds.begin(); iSeed!=inOutSeeds.end();iSeed++){
    
    std::cout << " InOutConversionTrackFinder::tracks hits in the seed " << iSeed->nHits() << std::endl;
    std::cout << " InOutConversionTrackFinder::tracks seed starting state position  " << iSeed->startingState().parameters().position() << " momentum " <<  iSeed->startingState().parameters().momentum() << " charge " << iSeed->startingState().parameters().charge() << std::endl;
    std::cout << " InOutConversionTrackFinder::tracks seed  starting state para, vector  " << iSeed->startingState().parameters().vector() << std::endl;
    
    
    
    std::vector<Trajectory> theTmpTrajectories;

    theTmpTrajectories = theCkfTrajectoryBuilder_->trajectories(*iSeed);
    
    std:: cout << " InOutConversionTrackFinder::track returned " << theTmpTrajectories.size() << " trajectories" << std::endl;
    
    theTrajectoryCleaner_->clean(theTmpTrajectories);
    
    for(std::vector<Trajectory>::const_iterator it=theTmpTrajectories.begin();
	it!=theTmpTrajectories.end(); it++){
      if( it->isValid() ) {
	rawResult.push_back(*it);
      }
    }
    std::cout << " InOutConversionTrackFinder::track rawResult size after cleaning " << rawResult.size() << std::endl;

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
  
    
  
  std::cout << " InOutConversionTrackFinder  tmpO size " << tmpO.size() << " before sorting " << std::endl; 
  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
    std::cout << " InOutConversionTrackFinder  tmpO num of hits " << (*it).foundHits() << " before ordering " << std::endl; 
    
  }
  
  precomputed_value_sort( tmpO.begin(), tmpO.end(), ExtractNumOfHits()  ); 
  
  
  std::cout << " InOutConversionTrackFinder  tmpO after sorting " << std::endl; 
  for (std::vector<Trajectory>::const_iterator it =tmpO.begin(); it != tmpO.end(); it++) {
    std::cout << " InOutConversionTrackFinder  tmpO  num of hits " << (*it).foundHits() << std::endl; 

  }
  
  




  
  std::cout << "  Returning " << result.size() << " In Out Tracks " << std::endl;
  return result;
  
  
}
