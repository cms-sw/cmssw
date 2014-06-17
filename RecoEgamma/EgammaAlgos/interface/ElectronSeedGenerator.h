#ifndef ElectronSeedGenerator_H
#define ElectronSeedGenerator_H

/** \class ElectronSeedGenerator

 * Class to generate the trajectory seed from two hits in
 *  the pixel detector which have been found compatible with
 *  an ECAL cluster.
 *
 * \author U.Berthon, C.Charlot, LLR Palaiseau
 *
 * \version   1st Version May 30, 2006
 *
 ************************************************************/

#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/EgammaReco/interface//ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
//#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"


class PropagatorWithMaterial;
class KFUpdator;
class PixelHitMatcher;
class MeasurementTracker;
class NavigationSchool;
class TrackerTopology;

class ElectronSeedGenerator
{
 public:

  struct Tokens {
    edm::EDGetTokenT<std::vector<reco::Vertex> > token_vtx;
    edm::EDGetTokenT<reco::BeamSpot> token_bs;
    edm::EDGetTokenT<MeasurementTrackerEvent> token_measTrkEvt;
  };

  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;

  ElectronSeedGenerator(const edm::ParameterSet&,
			const Tokens&);
  ~ElectronSeedGenerator();

  void setupES(const edm::EventSetup& setup);
  void run(
    edm::Event&, const edm::EventSetup& setup,
    const reco::SuperClusterRefVector &, const std::vector<float> & hoe1s, const std::vector<float> & hoe2s,
    TrajectorySeedCollection *seeds, reco::ElectronSeedCollection&);

 private:

  void seedsFromThisCluster( edm::Ref<reco::SuperClusterCollection> seedCluster, float hoe1, float hoe2, reco::ElectronSeedCollection & out, const TrackerTopology *tTopo ) ;
  void seedsFromRecHits( std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > & elePixelHits, PropagationDirection & dir, const GlobalPoint & vertexPos, const reco::ElectronSeed::CaloClusterRef & cluster, reco::ElectronSeedCollection & out, bool positron ) ;
  void seedsFromTrajectorySeeds( const std::vector<SeedWithInfo> & elePixelSeeds, const reco::ElectronSeed::CaloClusterRef & cluster, float hoe1, float hoe2, reco::ElectronSeedCollection & out, bool positron ) ;
  void addSeed( reco::ElectronSeed & seed, const SeedWithInfo * info, bool positron, reco::ElectronSeedCollection & out ) ;
  bool prepareElTrackSeed( ConstRecHitPointer outerhit,ConstRecHitPointer innerhit, const GlobalPoint & vertexPos) ;

  bool dynamicphiroad_;
  bool fromTrackerSeeds_;
  //  edm::EDGetTokenT<SomeClass> initialSeeds_;
  bool useRecoVertex_;
  edm::Handle<std::vector<reco::Vertex> > theVertices;
  edm::EDGetTokenT<std::vector<reco::Vertex> > verticesTag_;

  edm::Handle<reco::BeamSpot> theBeamSpot;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;

  float lowPtThreshold_;
  float highPtThreshold_;
  float nSigmasDeltaZ1_; // first z window size if not using the reco vertex
  float deltaZ1WithVertex_; // first z window size when using the reco vertex
  float sizeWindowENeg_;
  float phiMin2B_ ;
  float phiMax2B_ ;
  float phiMin2F_ ;
  float phiMax2F_ ;
  float deltaPhi1Low_, deltaPhi1High_;
  float deltaPhi2B_;
  float deltaPhi2F_;

  // so that deltaPhi1 = deltaPhi1Coef1_ + deltaPhi1Coef2_/clusterEnergyT
  double deltaPhi1Coef1_ ;
  double deltaPhi1Coef2_ ;

  PixelHitMatcher *myMatchEle;
  PixelHitMatcher *myMatchPos;

  //  edm::Handle<TrajectorySeedCollection> theInitialSeedColl;
  TrajectorySeedCollection* theInitialSeedColl;

  edm::ESHandle<MagneticField>                theMagField;
  edm::ESHandle<TrackerGeometry>              theTrackerGeometry;
  //edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;
  KFUpdator * theUpdator;
  PropagatorWithMaterial * thePropagator;

  std::string theMeasurementTrackerName;
  const MeasurementTracker*     theMeasurementTracker;
  edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerEventTag;

  const NavigationSchool*       theNavigationSchool;

  const edm::EventSetup *theSetup;
  

  PRecHitContainer recHits_;
  PTrajectoryStateOnDet pts_;

  // keep cacheIds to get records only when necessary
  unsigned long long cacheIDMagField_;
//  unsigned long long cacheIDGeom_;
  unsigned long long cacheIDNavSchool_;
  unsigned long long cacheIDCkfComp_;
  unsigned long long cacheIDTrkGeom_;
};

#endif // ElectronSeedGenerator_H


