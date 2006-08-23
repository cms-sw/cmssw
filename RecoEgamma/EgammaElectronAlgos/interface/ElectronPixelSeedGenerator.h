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

using namespace reco;  //FIXME

class ElectronPixelSeedGenerator
{
public:

  //RC
  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
  

  enum mode{HLT, offline, unknown};  //to be used later

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
			  float izmax2
			  );

  ~ElectronPixelSeedGenerator();

  void setup(bool);
  void setupES(const edm::EventSetup& setup, const edm::ParameterSet& conf);
  void run(const edm::Event&, ElectronPixelSeedCollection&);

 private:

  void seedsFromThisCluster(edm::Ref<SuperClusterCollection> seedCluster, ElectronPixelSeedCollection & out);
  //RC void prepareElTrackSeed(const TSiPixelRecHit& outerhit,const TSiPixelRecHit& innerhit, const GlobalPoint& vertexPos);
  //void prepareElTrackSeed(ConstRecHitPointer outerhit,ConstRecHitPointer innerhit, const GlobalPoint& vertexPos);
  bool prepareElTrackSeed(ConstRecHitPointer outerhit,ConstRecHitPointer innerhit, const GlobalPoint& vertexPos);
  

  float ephimin1;
  float ephimax1;
  float pphimin1;
  float pphimax1;
  float pphimin2, pphimax2;
  float zmin1, zmax1, zmin2, zmax2;

  std::string moduleLabel_;
  std::string instanceName_;

  PixelHitMatcher *myMatchEle;
  PixelHitMatcher *myMatchPos;
  mode theMode_;

  edm::ESHandle<MagneticField>                theMagField;
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;
  KFUpdator * theUpdator;
  PropagatorWithMaterial * thePropagator;

  const MeasurementTracker*     theMeasurementTracker;
  const NavigationSchool*       theNavigationSchool;

  const edm::EventSetup *theSetup; 
  TrajectoryStateTransform transformer_; 
  //RC recHitContainer recHits_; 
  PRecHitContainer recHits_; 
  PTrajectoryStateOnDet* pts_; 

  /*   edm::ValidityInterval vMag; */
  /*   edm::ValidityInterval vTrackerDigi; */
  /*   edm::ValidityInterval vTrackerReco; */

};

#endif // ElectronPixelSeedGenerator_H


