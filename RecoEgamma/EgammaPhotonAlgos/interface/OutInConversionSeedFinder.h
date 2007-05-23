#ifndef RecoEGAMMA_ConversionSeed_OutInConversionSeedFinder_h
#define RecoEGAMMA_ConversionSeed_OutInConversionSeedFinder_h

/** \class OutInConversionSeedFinder
 **  
 **
 **  $Id: OutInConversionSeedFinder.h,v 1.5 2007/03/14 20:34:30 nancy Exp $ 
 **  $Date: 2007/03/14 20:34:30 $ 
 **  $Revision: 1.5 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include <string>
#include <vector>



class MagneticField;
class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class  LayerMeasurements;

class OutInConversionSeedFinder : public ConversionSeedFinder {


 private:
 
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;


  public :
    
  
    OutInConversionSeedFinder( const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker );
  
  
  virtual ~OutInConversionSeedFinder();
  
  
  
  virtual void  makeSeeds(const reco::BasicClusterCollection& allBc) const  ;


 private:
  

  std::pair<FreeTrajectoryState,bool> makeTrackState(int charge) const ;

  void fillClusterSeeds(const reco::BasicCluster* bc) const ;

  void startSeed(const FreeTrajectoryState &) const;
  void completeSeed(const TrajectoryMeasurement & m1,
			    FreeTrajectoryState & fts, 
			    const Propagator *, 
			    int layer) const  ;
  void createSeed(const TrajectoryMeasurement & m1,const TrajectoryMeasurement & m2) const;
  FreeTrajectoryState createSeedFTS(const TrajectoryMeasurement & m1, const TrajectoryMeasurement & m2) const;  
  GlobalPoint fixPointRadius(const TrajectoryMeasurement &) const;



  
  MeasurementEstimator * makeEstimator(const DetLayer *, float dphi) const ;

  private :
    
    float  the2ndHitdphi_;
  float   the2ndHitdzConst_;    
  float  the2ndHitdznSigma_; 
  mutable std::vector<TrajectoryMeasurement> theFirstMeasurements_;
  mutable int nSeedsPerBC_;
  int maxNumberOfOutInSeedsPerBC_;

  const LayerMeasurements*      theLayerMeasurements_;
  

};

#endif
