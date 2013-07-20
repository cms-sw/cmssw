#ifndef RecoEGAMMA_ConversionSeed_ConversionSeedFinder_h
#define RecoEGAMMA_ConversionSeed_ConversionSeedFinder_h
/** \class ConversionSeedFinder
 **  
 **
 **  $Id: ConversionSeedFinder.h,v 1.13 2011/07/28 19:37:35 vlimant Exp $ 
 **  $Date: 2011/07/28 19:37:35 $ 
 **  $Revision: 1.13 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"


#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

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
  ConversionSeedFinder( const edm::ParameterSet& config );
  
  virtual ~ConversionSeedFinder(){}

  
  virtual void makeSeeds(const edm::Handle<edm::View<reco::CaloCluster> > & allBc ) const  =0 ;
 


  TrajectorySeedCollection & seeds() {  return theSeeds_;}
  virtual void setCandidate( float e, GlobalPoint pos ) const {  theSCenergy_=e; theSCPosition_= pos; }			       
  std::vector<const DetLayer*> const & layerList() const { return theLayerList_;}
 
  
  void setMeasurementTracker(const MeasurementTracker* tracker) const { ; }
  const MeasurementTracker* getMeasurementTracker() const  {return  theMeasurementTracker_;}

  /// Initialize EventSetup objects at each event
  void setEventSetup( const edm::EventSetup& es ) ; 
  void setEvent( const edm::Event& e ) ; 

  void clear() {
    theSeeds_.clear();
  }

 protected:


  edm::ParameterSet conf_;
  void findLayers() const ;
  void findLayers(const FreeTrajectoryState & fts) const  ; 

  FreeTrajectoryState trackStateFromClusters ( int aCharge,
					       const GlobalPoint & gpOrigine, 
					       PropagationDirection dir, 
					       float scaleFactor ) const;
  

  void printLayer(int i) const ;

  
  mutable TrajectorySeedCollection theSeeds_;
  mutable GlobalPoint theSCPosition_;
    

  std::string theMeasurementTrackerName_;
  const MeasurementTracker*     theMeasurementTracker_;
  const TrackingGeometry* theTrackerGeom_;

 
  edm::ESHandle<MagneticField> theMF_;
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker_;


  KFUpdator                  theUpdator_;
  PropagationDirection       dir_; 
  mutable reco::CaloCluster*  theSC_;
  mutable float theSCenergy_;  

  
  mutable std::vector<const DetLayer *> theLayerList_ ;    
    
  mutable GlobalPoint theBCPosition_;
  mutable float       theBCEnergy_; 

  const Propagator*  thePropagatorAlongMomentum_;
  const Propagator*  thePropagatorOppositeToMomentum_;

  reco::BeamSpot theBeamSpot_;



};

#endif
