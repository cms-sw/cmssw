#ifndef RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h

/** \class ConversionTrackFinder
 **  
 **
 **  $Id: ConversionTrackFinder.h,v 1.9 2011/02/26 15:01:43 nancy Exp $ 
 **  $Date: 2011/02/26 15:01:43 $ 
 **  $Revision: 1.9 $
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
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"

// C/C++ headers
#include <string>
#include <vector>

class TransientInitialStateEstimator;
class ConversionTrackFinder {

 public:
  
  ConversionTrackFinder( const edm::EventSetup& es,
			 const edm::ParameterSet& config );
                       
  
  virtual ~ConversionTrackFinder();
 
  
  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds , TrackCandidateCollection &candidate) const =0;

  /// Initialize EventSetup objects at each event
  void setEventSetup( const edm::EventSetup& es ) ; 
  void setEvent(const  edm::Event& e ) ; 


 private:

   



 protected: 
  
  edm::ParameterSet conf_;
  const MagneticField* theMF_;

  std::string theMeasurementTrackerName_;
  const MeasurementTracker*     theMeasurementTracker_;
  const TrajectoryBuilder*  theCkfTrajectoryBuilder_;

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
