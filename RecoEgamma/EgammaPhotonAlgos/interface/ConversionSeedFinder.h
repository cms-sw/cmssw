#ifndef RecoEGAMMA_ConversionSeed_ConversionSeedFinder_h
#define RecoEGAMMA_ConversionSeed_ConversionSeedFinder_h
/** \class ConversionSeedFinder
 **  
 **
 **  $Id: ConversionSeedFinder.h,v 1.1 2006/06/09 15:50:14 nancy Exp $ 
 **  $Date: 2006/06/09 15:50:14 $ 
 **  $Revision: 1.1 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/


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

class ConversionSeedFinder {

 public:
  

  ConversionSeedFinder();
  ConversionSeedFinder( const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);
  

  
  virtual ~ConversionSeedFinder(){}

  
  virtual void makeSeeds(const reco::BasicClusterCollection& allBc ) const  =0 ;
  TrajectorySeedCollection seeds() {std::cout << " Returning  seeds " << std::endl; return theSeeds_;}
  virtual void setCandidate(reco::SuperCluster& sc ) const { theSC_=&sc; }			       
  vector<const DetLayer*> layerList() const { return theLayerList_;}
 
  
  void setMeasurementTracker(const MeasurementTracker* tracker) const { ; }
  const MeasurementTracker* getMeasurementTracker() const  {return  theMeasurementTracker_;}

 protected:
  
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

  
  mutable vector<const DetLayer *> theLayerList_ ;    
    
  mutable GlobalPoint theBCPosition_;
  mutable float       theBCEnergy_; 



};

#endif
