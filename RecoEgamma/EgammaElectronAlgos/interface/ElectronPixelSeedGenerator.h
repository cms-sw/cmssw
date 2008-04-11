#ifndef ElectronPixelSeedGenerator_H
#define ElectronPixelSeedGenerator_H

/** \class ElectronPixelSeedGenerator
 
 * Class to generate the trajectory seed from two hits in 
 *  the pixel detector which have been found compatible with
 *  an ECAL cluster. 
 *
 * \author U.Berthon, C.Charlot, LLR Palaiseau
 *
 * \version   1st Version May 30, 2006  
 *
 ************************************************************/

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"


class PropagatorWithMaterial;
class KFUpdator;
class PixelHitMatcher;
class MeasurementTracker;
class NavigationSchool;

class ElectronPixelSeedGenerator: public ElectronSeedGenerator
{
 public:

  //RC
  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
  
  ElectronPixelSeedGenerator(const edm::ParameterSet&);
  ~ElectronPixelSeedGenerator();

  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&, const edm::EventSetup& setup, const reco::SuperClusterRefVector &, reco::ElectronPixelSeedCollection&);

 private:

  void seedsFromThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster, reco::ElectronPixelSeedCollection & out);
  bool prepareElTrackSeed(ConstRecHitPointer outerhit,ConstRecHitPointer innerhit, const GlobalPoint& vertexPos);

  bool dynamicphiroad_;
  bool fromTrackerSeeds_;
  edm::InputTag initialSeeds_;

  float lowPtThreshold_;
  float highPtThreshold_;
  float sizeWindowENeg_;   
  float phimin2_,phimax2_;
  float deltaPhi1Low_, deltaPhi1High_;
  float deltaPhi2_;
  
  double zmin1_, zmax1_;
  math::XYZPoint BSPosition_;  

  PixelHitMatcher *myMatchEle;
  PixelHitMatcher *myMatchPos;

  edm::Handle<TrajectorySeedCollection> theInitialSeedColl;

  edm::ESHandle<MagneticField>                theMagField;
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;
  KFUpdator * theUpdator;
  PropagatorWithMaterial * thePropagator;

  const MeasurementTracker*     theMeasurementTracker;
  const NavigationSchool*       theNavigationSchool;

  const edm::EventSetup *theSetup; 
  TrajectoryStateTransform transformer_; 

  PRecHitContainer recHits_; 
  PTrajectoryStateOnDet* pts_; 

  // keep cacheIds to get records only when necessary
  unsigned long long cacheIDMagField_;
  unsigned long long cacheIDGeom_;
  unsigned long long cacheIDNavSchool_;
  unsigned long long cacheIDCkfComp_;
  unsigned long long cacheIDTrkGeom_;
};

#endif // ElectronPixelSeedGenerator_H


