#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
//
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
//
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
//
#include "DataFormats/Common/interface/OwnVector.h"
#include "Utilities/General/interface/precomputed_value_sort.h"


/*
#include "ElectronPhoton/ClusterTools/interface/EgammaVSuperCluster.h"
#include "ElectronPhoton/ClusterTools/interface/EgammaCandidate.h"
//
#include "CommonReco/PatternTools/interface/TTrack.h"
#include "CommonReco/PatternTools/interface/SeedGenerator.h"
#include "CommonReco/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "CommonReco/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"
#include "CommonReco/KalmanUpdators/interface/KFUpdator.h"
#include "CommonReco/TrackFitters/interface/KFTrajectorySmoother.h"
#include "CommonReco/TrackFitters/interface/KFFittingSmoother.h"
#include "CommonReco/PatternTools/interface/ConcreteRecTrack.h"
#include "CommonReco/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "CommonReco/PatternTools/interface/TrajectorySeed.h"
#include "CommonReco/PatternTools/interface/TrajectoryBuilder.h"
#include "CommonReco/PatternTools/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackerReco/GtfPattern/interface/CombinatorialTrajectoryBuilder.h"
#include "TrackerReco/GtfPattern/interface/RedundantSeedCleaner.h"

*/



OutInConversionTrackFinder::OutInConversionTrackFinder(const edm::EventSetup& es, const edm::ParameterSet& conf, const MagneticField* field,  const MeasurementTracker* theInputMeasurementTracker ) :  ConversionTrackFinder(  field, theInputMeasurementTracker), conf_(conf)
{
  std::cout << " OutInConversionTrackFinder CTOR  theMeasurementTracker_   " << theMeasurementTracker_<<  std::endl; 

  
  seedClean_ = conf_.getParameter<bool>("outInSeedCleaning");
  
  theInitialState_       = new TransientInitialStateEstimator( es);
 

  theCkfTrajectoryBuilder_ = new CkfTrajectoryBuilder(conf_,es,theMeasurementTracker_);
  theTrajectoryCleaner_ = new TrajectoryCleanerBySharedHits();
  

}


OutInConversionTrackFinder::~OutInConversionTrackFinder() {

  delete theCkfTrajectoryBuilder_;
  delete theUpdator_;
  delete theTrajectoryCleaner_;

}



std::vector<const Trajectory*>  OutInConversionTrackFinder::tracks(const TrajectorySeedCollection outInSeeds  )const  {

// TrackCandidateCollection  OutInConversionTrackFinder::tracks(const TrajectorySeedCollection outInSeeds  )const  {
  
  std::cout << " OutInConversionTrackFinder::tracks  " <<  outInSeeds.size() << " Out-In seeds " << endl;
  std::vector<const  Trajectory*> tmp;
  tmp.erase(tmp.begin(), tmp.end() ) ;
  std::vector<const Trajectory*> result;
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
    
    std::vector<Trajectory> unsmoothedResult;
    theTrajectoryCleaner_->clean(rawResult);
    
    for (std::vector<Trajectory>::const_iterator itraw = rawResult.begin(); itraw != rawResult.end(); itraw++) {
      if((*itraw).isValid()) {
          unsmoothedResult.push_back( *itraw);
          tmp.push_back( &(*itraw) );
      }
    }
    



     
  }



 
  //  std::sort( tmp.begin(), tmp.end(), ByNumOfHits() ); 
 // precomputed_value_sort( tmp.begin(), tmp.end(), ExtractNumOfHits<Trajectory> ); 

  if ( tmp.size() ) {
    vector<const Trajectory*>::iterator it=tmp.begin();
    // only send out the two best tracks 
    result.push_back(*it);      
    if ( tmp.size() > 1) result.push_back(*(++it));
  }


  std::cout << "  Returning " << tmp.size() << " Out In Tracks " << std::endl;
  return result;

}
