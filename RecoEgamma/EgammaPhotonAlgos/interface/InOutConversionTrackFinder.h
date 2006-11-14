#ifndef RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
/** \class InOutConversionTrackFinder
 **  
 **
 **  $Id: InOutConversionTrackFinder.h,v 1.3 2006/07/10 17:56:45 nancy Exp $ 
 **  $Date: 2006/07/10 17:56:45 $ 
 **  $Revision: 1.3 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"

// C/C++ headers
#include <string>
#include <vector>

//

class MagneticField;
class TrackerTrajectoryBuilder;
class TrajectoryCleanerBySharedHits;
class TransientInitialStateEstimator;



class InOutConversionTrackFinder : public ConversionTrackFinder {
 public :
   
   InOutConversionTrackFinder(  const edm::EventSetup& es,const edm::ParameterSet& config, const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);
 
 virtual ~InOutConversionTrackFinder();
  

 virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds ) const ;
 // virtual std::vector<const Trajectory*> tracks(const TrajectorySeedCollection seeds ) const;
 
 
 private:
 
 edm::ParameterSet conf_;
 void initComponents();
 const TrackerTrajectoryBuilder*  theCkfTrajectoryBuilder_;
 TrajectoryCleanerBySharedHits* theTrajectoryCleaner_;
 
 TransientInitialStateEstimator* theInitialState_; 

 const TrackerGeometry* trackerGeom;


};

#endif
