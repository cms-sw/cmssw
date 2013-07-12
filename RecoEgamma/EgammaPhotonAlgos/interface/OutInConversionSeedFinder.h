#ifndef RecoEGAMMA_ConversionSeed_OutInConversionSeedFinder_h
#define RecoEGAMMA_ConversionSeed_OutInConversionSeedFinder_h

/** \class OutInConversionSeedFinder
 **  
 **
 **  $Id: OutInConversionSeedFinder.h,v 1.8 2008/05/08 20:41:27 nancy Exp $ 
 **  $Date: 2008/05/08 20:41:27 $ 
 **  $Revision: 1.8 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

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
    
  
    OutInConversionSeedFinder( const edm::ParameterSet& config );
  
  virtual ~OutInConversionSeedFinder();
  
  
  
  virtual void  makeSeeds( const edm::Handle<edm::View<reco::CaloCluster> > & allBc) const  ;
  virtual void  makeSeeds( const reco::CaloClusterPtr&  aBC ) const  ;  
  
 private:
  
  edm::ParameterSet conf_;
  std::pair<FreeTrajectoryState,bool> makeTrackState(int charge) const ;
  
  void fillClusterSeeds(const reco::CaloClusterPtr& bc) const ;
  
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


};

#endif
