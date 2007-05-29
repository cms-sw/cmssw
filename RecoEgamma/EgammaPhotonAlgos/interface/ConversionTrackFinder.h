#ifndef RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h

/** \class ConversionTrackFinder
 **  
 **
 **  $Id: ConversionTrackFinder.h,v 1.5 2007/02/19 20:47:37 nancy Exp $ 
 **  $Date: 2007/02/19 20:47:37 $ 
 **  $Revision: 1.5 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
//
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
//#include "DataFormats/TrackCandidate/interface/TrackCandidateSuperClusterAssociation.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

// C/C++ headers
#include <string>
#include <vector>


class ConversionTrackFinder {

 public:
  
  ConversionTrackFinder(const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker)  :  theMF_(field),  theMeasurementTracker_(theInputMeasurementTracker)
    {
    }

  
  virtual ~ConversionTrackFinder()
    {
    }

  
  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds , TrackCandidateCollection &candidate) const =0;
  

 protected: 
  

  const MagneticField* theMF_;
  const MeasurementTracker*     theMeasurementTracker_;
  bool seedClean_;
  double smootherChiSquare_;


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
