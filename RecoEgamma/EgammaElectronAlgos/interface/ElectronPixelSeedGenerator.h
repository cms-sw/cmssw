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

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"  
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"  
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ValidityInterval.h"

class PropagatorWithMaterial;
class KFUpdator;
class PixelHitMatcher;
class MeasurementTracker;
class NavigationSchool;

class ElectronPixelSeedGenerator
{
public:

  //RC
  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
  
  ElectronPixelSeedGenerator(
                          float iephimin1,
			  float iephimax1,
			  float ipphimin1,
			  float ipphimax1,
			  float iphimin2,
			  float iphimax2,
			  float izmin1,
			  float izmax1,
			  float izmin2,
			  float izmax2,
			  bool  idynamicphiroad
			  );

  ~ElectronPixelSeedGenerator();

  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&, const edm::EventSetup& setup, const edm::Handle<reco::SuperClusterCollection>&, reco::ElectronPixelSeedCollection&);

 private:

  void seedsFromThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster, reco::ElectronPixelSeedCollection & out);
  bool prepareElTrackSeed(ConstRecHitPointer outerhit,ConstRecHitPointer innerhit, const GlobalPoint& vertexPos);

  float ephimin1;
  float ephimax1;
  float pphimin1;
  float pphimax1;
  float pphimin2, pphimax2;
  float zmin1, zmax1, zmin2, zmax2;
  bool dynamicphiroad;
  
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


