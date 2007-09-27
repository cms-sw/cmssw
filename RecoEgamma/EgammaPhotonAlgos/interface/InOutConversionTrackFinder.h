#ifndef RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
/** \class InOutConversionTrackFinder
 **  
 **
 **  $Id: InOutConversionTrackFinder.h,v 1.6 2007/06/25 16:39:36 nancy Exp $ 
 **  $Date: 2007/06/25 16:39:36 $ 
 **  $Revision: 1.6 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
//
#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"

// C/C++ headers
#include <string>
#include <vector>

//

class MagneticField;
class TrajectoryBuilder;
class TrajectoryCleanerBySharedHits;
class TransientInitialStateEstimator;



class InOutConversionTrackFinder : public ConversionTrackFinder {
 public :
   
   InOutConversionTrackFinder(  const edm::EventSetup& es,const edm::ParameterSet& config, const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);
 
 virtual ~InOutConversionTrackFinder();
 virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds, TrackCandidateCollection &candidate ) const ;

 
 private:
 
 edm::ParameterSet conf_;
 void initComponents();
 const TrajectoryBuilder*  theCkfTrajectoryBuilder_;
 TrajectoryCleanerBySharedHits* theTrajectoryCleaner_;
 
 TransientInitialStateEstimator* theInitialState_; 

 const TrackerGeometry* trackerGeom;

 RedundantSeedCleaner*  theSeedCleaner_;


};

#endif
