#ifndef SubSeedGenerator_H
#define SubSeedGenerator_H

/** \class SubSeedGenerator
 *
 ************************************************************/
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"

#include <TMath.h>

#include <Math/VectorUtil.h>
#include <Math/Point3D.h>

class SubSeedGenerator : public ElectronSeedGenerator
{
public:

  
  SubSeedGenerator(const std::string &seedProducer, const std::string &seedLabel);

  ~SubSeedGenerator();

  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&, const edm::EventSetup& setup, const edm::Handle<reco::SuperClusterCollection>&, reco::ElectronPixelSeedCollection&);

 private:

/*   void seedsFromThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster, reco::ElectronPixelSeedCollection & out); */
/*   bool prepareElTrackSeed(ConstRecHitPointer outerhit,ConstRecHitPointer innerhit, const GlobalPoint& vertexPos); */

/*   float ephimin1; */
/*   float ephimax1; */
/*   float pphimin1; */
/*   float pphimax1; */
/*   float pphimin2, pphimax2; */
/*   float zmin1, zmax1, zmin2, zmax2; */
/*   bool dynamicphiroad; */
  
/*   PixelHitMatcher *myMatchEle; */
/*   PixelHitMatcher *myMatchPos; */

/*   edm::ESHandle<MagneticField>                theMagField; */
/*   edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker; */
/*   KFUpdator * theUpdator; */
/*   PropagatorWithMaterial * thePropagator; */

/*   const MeasurementTracker*     theMeasurementTracker; */
/*   const NavigationSchool*       theNavigationSchool; */

/*   const edm::EventSetup *theSetup;  */
/*   TrajectoryStateTransform transformer_;  */

/*   PRecHitContainer recHits_;  */
/*   PTrajectoryStateOnDet* pts_;  */
  std::string initialSeedProducer_;
  std::string initialSeedLabel_;
};

#endif // SubSeedGenerator_H


