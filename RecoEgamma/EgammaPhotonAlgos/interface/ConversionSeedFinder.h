#ifndef RecoEGAMMA_ConversionSeed_ConversionSeedFinder_h
#define RecoEGAMMA_ConversionSeed_ConversionSeedFinder_h
/** \class ConversionSeedFinder
 **  
 **
 **  $Id: ConversionSeedFinder.h,v 1.5 2007/02/05 13:23:16 nancy Exp $ 
 **  $Date: 2007/02/05 13:23:16 $ 
 **  $Revision: 1.5 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"


// C/C++ headers
#include <string>
#include <vector>




//


class DetLayer;
class FreeTrajectoryState;
class TrajectoryStateOnSurface;


class ConversionSeedFinder {

 public:
  

  ConversionSeedFinder();
  ConversionSeedFinder(  const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);

  

  
  virtual ~ConversionSeedFinder(){}

  
  virtual void makeSeeds(const reco::BasicClusterCollection& allBc ) const  =0 ;
 


  TrajectorySeedCollection seeds() {  return theSeeds_;}
  virtual void setCandidate(reco::SuperCluster& sc ) const { theSC_=&sc; }			       
  std::vector<const DetLayer*> layerList() const { return theLayerList_;}
 
  
  void setMeasurementTracker(const MeasurementTracker* tracker) const { ; }
  const MeasurementTracker* getMeasurementTracker() const  {return  theMeasurementTracker_;}

 protected:


  void findLayers() const ;
  void findLayers(const FreeTrajectoryState & fts) const  ; 

  FreeTrajectoryState trackStateFromClusters ( int aCharge,
					       const GlobalPoint & gpOrigine, 
					       PropagationDirection dir, 
					       float scaleFactor ) const;
  

  void printLayer(int i) const ;

  
  mutable TrajectorySeedCollection theSeeds_;
  mutable GlobalPoint theSCPosition_;
    
  const MagneticField* theMF_;
  const  GeometricSearchTracker*  theTracker_;
  const MeasurementTracker*     theMeasurementTracker_;
  
  StraightLinePropagator     theOutwardStraightPropagator_;		       
  mutable PropagatorWithMaterial     thePropagatorWithMaterial_; 
  KFUpdator                  theUpdator_;
  PropagationDirection       dir_; 
  mutable reco::SuperCluster*  theSC_;

  
  mutable std::vector<const DetLayer *> theLayerList_ ;    
    
  mutable GlobalPoint theBCPosition_;
  mutable float       theBCEnergy_; 



};

#endif
