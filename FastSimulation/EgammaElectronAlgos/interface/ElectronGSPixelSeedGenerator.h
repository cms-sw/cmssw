#ifndef ElectronGSPixelSeedGenerator_H
#define ElectronGSPixelSeedGenerator_H

/** \class ElectronGSPixelSeedGenerator
 
 * Class to generate the trajectory seed from two hits in 
 *  the pixel detector which have been found compatible with
 *  an ECAL cluster. 
 *
 * \author Patrick Janot
 *
 ************************************************************/

#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"  
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"

namespace edm { 
  class EventSetup;
  class ParameterSet;
  class Event;
}

class TrackerGeometry;
class MagneticField;
class GeometricSearchTracker;

class PropagatorWithMaterial;
class KFUpdator;
class GSPixelHitMatcher;

class ElectronGSPixelSeedGenerator
{
public:

  //RC
  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
  

  enum mode{HLT, offline, unknown};  //to be used later

  ElectronGSPixelSeedGenerator(
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

  ~ElectronGSPixelSeedGenerator();

  void setup(bool);

  void setupES(const edm::EventSetup& setup);

  void run(edm::Event&, 
	   const edm::Handle<reco::SuperClusterCollection>&, 
	   reco::ElectronPixelSeedCollection&);

 private:

  void addASeedToThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster,
			     std::vector<ConstRecHitPointer>&,
			     std::vector<reco::ElectronPixelSeed>&);

  bool prepareElTrackSeed(ConstRecHitPointer outerhit,
			  ConstRecHitPointer innerhit, 
			  const GlobalPoint& vertexPos);

  float ephimin1;
  float ephimax1;
  float pphimin1;
  float pphimax1;
  float phimin2, phimax2;
  float zmin1, zmax1, zmin2, zmax2;
  bool dynamicphiroad;
  
  GSPixelHitMatcher *myGSPixelMatcher;
  mode theMode_;

  const MagneticField* theMagField;
  const TrackerGeometry*  theTrackerGeometry;
  const GeometricSearchTracker* theGeomSearchTracker;

  KFUpdator * theUpdator;
  PropagatorWithMaterial * thePropagator;

  const edm::EventSetup *theSetup; 
  TrajectoryStateTransform transformer_; 
  PRecHitContainer recHits_; 
  PTrajectoryStateOnDet* pts_; 

};

#endif // ElectronGSPixelSeedGenerator_H


