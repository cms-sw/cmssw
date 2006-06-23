#ifndef RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
/** \class OutInConversionTrackFinder
 **  
 **
 **  $Id: OutInConversionTrackFinder.h,v 1.1 2006/06/09 15:51:32 nancy Exp $ 
 **  $Date: 2006/06/09 15:51:32 $ 
 **  $Revision: 1.1 $
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
class CkfTrajectoryBuilder;
class KFUpdator;
class TrajectoryCleanerBySharedHits;
class TransientInitialStateEstimator;

class OutInConversionTrackFinder : public ConversionTrackFinder {
 


  public :
    
    OutInConversionTrackFinder( const edm::EventSetup& es,  const edm::ParameterSet& config, const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);
  
  
  virtual ~OutInConversionTrackFinder();
   
  
  
  virtual std::vector<const Trajectory*> tracks(const TrajectorySeedCollection seeds ) const;


  //  virtual  TrackCandidateCollection  tracks(const TrajectorySeedCollection seeds ) const;
  
  
  
  

 private: 

  edm::ParameterSet                         conf_;
  CkfTrajectoryBuilder*  theCkfTrajectoryBuilder_;
  KFUpdator*                          theUpdator_;
  TrajectoryCleanerBySharedHits* theTrajectoryCleaner_;
  mutable std::vector<const Trajectory*>  theOutInTracks_;
 
  TransientInitialStateEstimator*theInitialState_;  



class ByNumOfHits {
 public:
  bool operator()(const Trajectory * a, const Trajectory * b) {
    if (a->foundHits() == b->foundHits()) {
      return a->chiSquared() < b->chiSquared();
    } else {
      return a->foundHits() > b->foundHits();
    }
  }
};




/*
template <class T, class Scalar = typename T::Scalar>
  struct ExtractNumOfHits {
    typedef Scalar result_type;
    Scalar operator()(const T* p) const {return p->foundHits();}
    Scalar operator()(const T& p) const {return p.foundHits();}
  };
*/



};






#endif
