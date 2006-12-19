#ifndef RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
/** \class InOutConversionTrackFinder
 **  
 **
 **  $Id: InOutConversionTrackFinder.h,v 1.4 2006/11/14 11:55:11 nancy Exp $ 
 **  $Date: 2006/11/14 11:55:11 $ 
 **  $Revision: 1.4 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
//#include "DataFormats/TrackCandidate/interface/TrackCandidateSuperClusterAssociation.h"
//
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
  

 //virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds, TrackCandidateCollection &candidates, reco::TrackCandidateSuperClusterAssociationCollection& outAssoc, int iSC ) const;
  

  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds, TrackCandidateCollection &candidate ) const ;

 //virtual  std::auto_ptr<TrackCandidateCollection>  tracks(const TrajectorySeedCollection seeds ) const;
// virtual TrackCandidateCollection tracks(const TrajectorySeedCollection seeds ) const ;
 // virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds ) const ;
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
