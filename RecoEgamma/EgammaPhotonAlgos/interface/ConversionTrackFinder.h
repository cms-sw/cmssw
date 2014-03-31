#ifndef RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h

/** \class ConversionTrackFinder
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
//
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

// C/C++ headers
#include <string>
#include <vector>

class TransientInitialStateEstimator;
class ConversionTrackFinder {

 public:
  
  ConversionTrackFinder( const edm::EventSetup& es,
			 const edm::ParameterSet& config );
                       
  
  virtual ~ConversionTrackFinder();
 
  
  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection& seeds , TrackCandidateCollection &candidate) const =0;

  /// Initialize EventSetup objects at each event
  void setEventSetup( const edm::EventSetup& es ) ; 
  void setTrajectoryBuilder(const BaseCkfTrajectoryBuilder & builder) ; 


 private:

   



 protected: 
  
  edm::ParameterSet conf_;
  const MagneticField* theMF_;

  std::string theMeasurementTrackerName_;
  const MeasurementTracker*     theMeasurementTracker_;
  const BaseCkfTrajectoryBuilder*  theCkfTrajectoryBuilder_;

  TransientInitialStateEstimator* theInitialState_;  
  const TrackerGeometry* theTrackerGeom_;
  KFUpdator*                          theUpdator_;

  edm::ESHandle<Propagator> thePropagator_;

  bool useSplitHits_;

struct ExtractNumOfHits {
  typedef int result_type;
  result_type operator()(const Trajectory& t) const {return t.foundHits();}
  result_type operator()(const Trajectory* t) const {return t->foundHits();}
};


struct ExtractChi2 {
  typedef float result_type;
  result_type operator()(const Trajectory& t) const {return t.chiSquared();}
  result_type operator()(const Trajectory* t) const {return t->chiSquared();}
};


 

};

#endif
