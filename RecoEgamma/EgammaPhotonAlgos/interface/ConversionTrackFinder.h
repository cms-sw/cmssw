#ifndef RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h

/** \class ConversionTrackFinder
 **  
 **
 **  $Id: ConversionTrackFinder.h,v 1.3 2006/07/10 17:55:25 nancy Exp $ 
 **  $Date: 2006/07/10 17:55:25 $ 
 **  $Revision: 1.3 $
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
      std::cout << " ConversionTrackFinder CTOR  theMeasurementTracker_ " <<  theMeasurementTracker_ << std:: endl;      
    }

  
  virtual ~ConversionTrackFinder()
    {
    }


  //  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds, TrackCandidateCollection &candidates, reco::TrackCandidateSuperClusterAssociationCollection& outAssoc, int iSC ) const =0;
  
  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds , TrackCandidateCollection &candidate) const =0;
  
  //virtual std::vector<const Trajectory*> tracks(const TrajectorySeedCollection seeds ) const =0;
  //virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds ) const =0;
  //virtual  TrackCandidateCollection tracks(const TrajectorySeedCollection seeds ) const =0;
  //  virtual std::auto_ptr<TrackCandidateCollection> tracks(const TrajectorySeedCollection seeds ) const =0;


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
