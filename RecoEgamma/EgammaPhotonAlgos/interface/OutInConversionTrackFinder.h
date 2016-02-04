#ifndef RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
/** \class OutInConversionTrackFinder
 **  
 **
 **  $Id: OutInConversionTrackFinder.h,v 1.9 2008/05/08 20:41:27 nancy Exp $ 
 **  $Date: 2008/05/08 20:41:27 $ 
 **  $Revision: 1.9 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
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
class TrajectoryCleanerBySharedHits;

class OutInConversionTrackFinder : public ConversionTrackFinder {
 


  public :
    
    OutInConversionTrackFinder( const edm::EventSetup& es,  
				const edm::ParameterSet& config );
  
  
  virtual ~OutInConversionTrackFinder();
  
  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds, TrackCandidateCollection &candidates ) const;

 private: 


 
  TrajectoryCleanerBySharedHits* theTrajectoryCleaner_;
  RedundantSeedCleaner*  theSeedCleaner_;


};






#endif
