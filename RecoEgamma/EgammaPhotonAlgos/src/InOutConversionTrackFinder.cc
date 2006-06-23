//
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
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
//
#include "CARF/Reco/interface/ParameterSetBuilder.h"
#include "CARF/Reco/interface/ParameterSet.h"
#include "CARF/Reco/interface/ComponentSetBuilder.h"
#include "CARF/Reco/interface/ComponentSet.h"
#include "CARF/Reco/interface/ConfigAlgoFactory.h"
#include "CARF/Reco/interface/RecQuery.h"

*/

InOutConversionTrackFinder::InOutConversionTrackFinder(const edm::EventSetup& es, const edm::ParameterSet& conf, const MagneticField* field,  const MeasurementTracker* theInputMeasurementTracker ) :  ConversionTrackFinder( field, theInputMeasurementTracker) , conf_(conf) {
  std::cout << " InOutConversionTrackFinder CTOR " << std:: endl;  
    
  seedClean_ = conf_.getParameter<bool>("inOutSeedCleaning");
  smootherChiSquare_ = conf_.getParameter<double>("smootherChiSquareCut");   



}




std::vector<const Trajectory*>  InOutConversionTrackFinder::tracks(const TrajectorySeedCollection seeds )const  {
// TrackCandidateCollection InOutConversionTrackFinder::tracks(const TrajectorySeedCollection seeds )const  {

  
  std::cout << "  Returning " << theInOutTracks_.size() << " In Out Tracks " << std::endl;
  return theInOutTracks_;

 }
