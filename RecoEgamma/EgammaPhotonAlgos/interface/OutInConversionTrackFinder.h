#ifndef RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
/** \class OutInConversionTrackFinder
 **  
 **
 **  $Id: OutInConversionTrackFinder.h,v 1.3 2006/07/10 17:57:18 nancy Exp $ 
 **  $Date: 2006/07/10 17:57:18 $ 
 **  $Revision: 1.3 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

// C/C++ headers
#include <string>
#include <vector>

//

class MagneticField;
class TrackerTrajectoryBuilder;
class KFUpdator;
class TrajectoryCleanerBySharedHits;
class TransientInitialStateEstimator;

class OutInConversionTrackFinder : public ConversionTrackFinder {
 


  public :
    
    OutInConversionTrackFinder( const edm::EventSetup& es,  const edm::ParameterSet& config, const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);
  
  
  virtual ~OutInConversionTrackFinder();
   
  
  
  // virtual std::vector<const Trajectory*> tracks(const TrajectorySeedCollection seeds ) const ;
  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds ) const;



  
  
  
  

 private: 

  edm::ParameterSet                         conf_;
  const TrackerTrajectoryBuilder*  theCkfTrajectoryBuilder_;
  KFUpdator*                          theUpdator_;
  TrajectoryCleanerBySharedHits* theTrajectoryCleaner_;
 
 
  TransientInitialStateEstimator* theInitialState_;  



};






#endif
