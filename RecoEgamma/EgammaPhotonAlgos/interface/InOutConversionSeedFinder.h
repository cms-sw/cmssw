#ifndef RecoEGAMMA_ConversionSeed_InOutConversionSeedFinder_h
#define RecoEGAMMA_ConversionSeed_InOutConversionSeedFinder_h

/** \class InOutConversionSeedFinder
 **  
 **
 **  $Id: InOutConversionSeedFinder.h,v 1.5 2006/12/19 17:35:31 nancy Exp $ 
 **  $Date: 2006/12/19 17:35:31 $ 
 **  $Revision: 1.5 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"

#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include <string>
#include <vector>



class MagneticField;
class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class TrajectoryMeasurement;

class InOutConversionSeedFinder : public ConversionSeedFinder {


 private:
 
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;


  public :
    
  
  
  InOutConversionSeedFinder( const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);
    
     
  virtual ~InOutConversionSeedFinder();
    

  virtual void  makeSeeds( const reco::BasicClusterCollection& allBc) const;
 

  //  void setTracks(std::vector<Trajectory> in) { inputTracks_.clear(); inputTracks_ = in;}
  void setTracks(std::vector<Trajectory> in) { theOutInTracks_.clear(); theOutInTracks_ = in;}
 

  private :


  virtual void fillClusterSeeds(  ) const ;
  void startSeed(FreeTrajectoryState * fts, const TrajectoryStateOnSurface & stateAtPreviousLayer, int charge, int layer) const ;
  virtual void findSeeds(const TrajectoryStateOnSurface & startingState,
			 float signedpt, unsigned int startingLayer) const ;
   
  std::vector<const reco::BasicCluster*> getSecondBasicClusters(const GlobalPoint & conversionPosition, float charge) const;
  void completeSeed(const TrajectoryMeasurement & m1,FreeTrajectoryState & fts, const Propagator* propagator, int ilayer) const;
  void createSeed(const TrajectoryMeasurement & m1,  const TrajectoryMeasurement & m2) const ;


 private:
  float  the2ndHitdphi_;
  float   the2ndHitdzConst_;    
  float  the2ndHitdznSigma_;
  mutable int track2Charge_;
  mutable GlobalVector track2InitialMomentum_; 
     


  
  std::vector<Trajectory> inputTracks_;
  std::vector<Trajectory> theOutInTracks_;
  mutable std::vector<TrajectoryMeasurement> theFirstMeasurements_;
  
  const LayerMeasurements*      theLayerMeasurements_;
  mutable reco::BasicCluster theSecondBC_;
  mutable reco::BasicClusterCollection  bcCollection_;

  
};

#endif
