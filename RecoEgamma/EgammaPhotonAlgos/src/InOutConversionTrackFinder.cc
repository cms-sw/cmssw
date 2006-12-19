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

  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  trackerGeom= trackerHandle.product();



  theTrajectoryCleaner_ = new TrajectoryCleanerBySharedHits();


}


InOutConversionTrackFinder::~InOutConversionTrackFinder() {

  //  delete theCkfTrajectoryBuilder_;
  delete theTrajectoryCleaner_;
  delete theInitialState_;
}




//std::auto_ptr<TrackCandidateCollection>  InOutConversionTrackFinder::tracks(const TrajectorySeedCollection inOutSeeds )const  {
//std::vector<Trajectory> InOutConversionTrackFinder::tracks(const TrajectorySeedCollection  inOutSeeds, 
//                                                          TrackCandidateCollection &output_p,    
//                                                           reco::TrackCandidateSuperClusterAssociationCollection& outAssoc, int iSC )const  {


std::vector<Trajectory> InOutConversionTrackFinder::tracks(const TrajectorySeedCollection  inOutSeeds, 
                                                           TrackCandidateCollection &output_p ) const {



  std::cout << " InOutConversionTrackFinder::tracks getting " <<  inOutSeeds.size() << " In-Out seeds " << endl;
  
  std::vector<Trajectory> tmpO;
  tmpO.erase(tmpO.begin(), tmpO.end() ) ;
  
  std::vector<Trajectory> result;
  result.erase(result.begin(), result.end() ) ;


  std::vector<Trajectory> rawResult;
  rawResult.erase(rawResult.begin(), rawResult.end() ) ;



  // Loop over the seeds
  for(TrajectorySeedCollection::const_iterator iSeed=inOutSeeds.begin(); iSeed!=inOutSeeds.end();iSeed++){

    /*    
    std::cout << " InOutConversionTrackFinder::tracks hits in the seed " << iSeed->nHits() << std::endl;
    std::cout << " InOutConversionTrackFinder::tracks seed starting state position  " << iSeed->startingState().parameters().position() << " momentum " <<  iSeed->startingState().parameters().momentum() << " charge " << iSeed->startingState().parameters().charge() << std::endl;
    std::cout << " InOutConversionTrackFinder::tracks seed  starting state para, vector  " << iSeed->startingState().parameters().vector() << std::endl;
    */

    
    std::vector<Trajectory> theTmpTrajectories;

    theTmpTrajectories = theCkfTrajectoryBuilder_->trajectories(*iSeed);
    
    std:: cout << " InOutConversionTrackFinder::track returned " << theTmpTrajectories.size() << " trajectories for this seed " << std::endl;
    
     theTrajectoryCleaner_->clean(theTmpTrajectories);
    
    for(std::vector<Trajectory>::const_iterator it=theTmpTrajectories.begin(); it!=theTmpTrajectories.end(); it++){
      if( it->isValid() ) {
	rawResult.push_back(*it);
		
      }
    }

  }
   
 
  std::vector<Trajectory> unsmoothedResult;
  std::cout << " InOutConversionTrackFinder::track  Start second cleaning " << std::endl;
  theTrajectoryCleaner_->clean(rawResult);
  std::cout << " InOutConversionTrackFinder::track rawResult size after cleaning " << rawResult.size() << std::endl;
  


  int tra=0;
  for (std::vector<Trajectory>::const_iterator itraw = rawResult.begin(); itraw != rawResult.end(); itraw++) {
    tra++;
    std::cout << " looping of rawResult after cleaning " << tra << std::endl;
    if((*itraw).isValid()) {
      // unsmoothedResult.push_back( *itraw);
	  tmpO.push_back( *itraw );
	  std::cout << " InOutConversionTrackFinder::track rawResult num of valid recHits per trajectory " << (*itraw).foundHits() << std::endl;
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

  for (int i=tmpO.size()-1; i>=0; i--) {
    unsmoothedResult.push_back(  tmpO[i] );  
  }
  std::cout << " InOutConversionTrackFinder  unsmoothedResult size  " <<  unsmoothedResult.size() << std::endl;   

  for (std::vector<Trajectory>::const_iterator it =  unsmoothedResult.begin(); it !=  unsmoothedResult.end(); it++) {
    std::cout << " InOutConversionTrackFinder  unsmoothedResult  after reordering " <<(*it).foundHits() <<  std::endl; 

  }


  // Convert to TrackCandidates and fill in the output_p
  for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin(); it != unsmoothedResult.end(); it++) {
    
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
    
    output_p.push_back(TrackCandidate(recHits, it->seed(),*state ) );
    delete state;
  }
  
      



  

  std::cout << "  InOutConversionTrackFinder::track Returning " <<  unsmoothedResult.size() << " In Out Trajectories " << std::endl;
  return  unsmoothedResult;
  

  
}
