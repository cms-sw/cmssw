#ifndef RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
/** \class OutInConversionTrackFinder
 **  
 **
 **  $Id: OutInConversionTrackFinder.h,v 1.4 2006/11/14 11:56:37 nancy Exp $ 
 **  $Date: 2006/11/14 11:56:37 $ 
 **  $Revision: 1.4 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
//#include "DataFormats/TrackCandidate/interface/TrackCandidateSuperClusterAssociation.h"
//
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
//
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
   
  //  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds, TrackCandidateCollection &candidates, reco::TrackCandidateSuperClusterAssociationCollection& outAssoc, int iSC ) const;
  
 virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds, TrackCandidateCollection &candidates ) const;
  


  // virtual std::vector<const Trajectory*> tracks(const TrajectorySeedCollection seeds ) const ;
  //  virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection seeds ) const;
  // virtual TrackCandidateCollection tracks(const TrajectorySeedCollection seeds ) const;
  //virtual  std::auto_ptr<TrackCandidateCollection>  tracks(const TrajectorySeedCollection seeds ) const;

  
  
  
  

 private: 


  edm::ParameterSet                         conf_;
  const TrackerTrajectoryBuilder*  theCkfTrajectoryBuilder_;
  KFUpdator*                          theUpdator_;
  TrajectoryCleanerBySharedHits* theTrajectoryCleaner_;
 
 
  TransientInitialStateEstimator* theInitialState_;  



};






#endif
