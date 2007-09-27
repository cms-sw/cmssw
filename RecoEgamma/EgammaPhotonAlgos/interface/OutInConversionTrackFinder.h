#ifndef RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
/** \class OutInConversionTrackFinder
 **  
 **
 **  $Id: OutInConversionTrackFinder.h,v 1.6 2007/06/25 16:39:42 nancy Exp $ 
 **  $Date: 2007/06/25 16:39:42 $ 
 **  $Revision: 1.6 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
//
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

// C/C++ headers
#include <string>
#include <vector>

//

class MagneticField;
class TrajectoryBuilder;
class KFUpdator;
class TrajectoryCleanerBySharedHits;
class TransientInitialStateEstimator;

class OutInConversionTrackFinder : public ConversionTrackFinder {
 


  public :
    
    OutInConversionTrackFinder( const edm::EventSetup& es,  const edm::ParameterSet& config, const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);
  
  
  virtual ~OutInConversionTrackFinder();
  
  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds, TrackCandidateCollection &candidates ) const;

 private: 


  edm::ParameterSet                         conf_;
  const TrajectoryBuilder*  theCkfTrajectoryBuilder_;
  KFUpdator*                          theUpdator_;
  TrajectoryCleanerBySharedHits* theTrajectoryCleaner_;
 
 
  TransientInitialStateEstimator* theInitialState_;  

  RedundantSeedCleaner*  theSeedCleaner_;


};






#endif
