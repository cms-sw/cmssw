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
#include "DataFormats/Math/interface/Point3D.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


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
  void run(edm::Event&, const edm::EventSetup& setup, const edm::Handle<reco::SuperClusterCollection>&, reco::ElectronPixelSeedCollection&);

 private:

  void seedsFromThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster, reco::ElectronPixelSeedCollection & out);
  bool prepareElTrackSeed(ConstRecHitPointer outerhit,ConstRecHitPointer innerhit, const GlobalPoint& vertexPos);

  float fEtaBarrelBad(float scEta);
  float fEtaEndcapBad(float scEta);
  float fEtaBarrelGood(float scEta);
  float fEtaEndcapGood(float scEta);
  
  bool dynamicphiroad_;
  double SCEtCut_;
  float lowPtThreshold_;
  float highPtThreshold_;
  float sizeWindowENeg_;   
  float phimin2_,phimax2_;
  float deltaPhi1Low_, deltaPhi2Low_;
  float deltaPhi1High_, deltaPhi2High_;
  
  double zmin1_, zmax1_;
  math::XYZPoint BSPosition_;  

  PixelHitMatcher *myMatchEle;
  PixelHitMatcher *myMatchPos;

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
};

#endif // ElectronPixelSeedGenerator_H


