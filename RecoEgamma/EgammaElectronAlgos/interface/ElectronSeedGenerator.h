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

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
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

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

class PropagatorWithMaterial;
class KFUpdator;
class PixelHitMatcher_;
class MeasurementTracker;
class NavigationSchool;
class TrackerTopology;

class ElectronSeedGenerator {
public:
  struct Tokens {
    edm::EDGetTokenT<std::vector<reco::Vertex> > token_vtx;
    edm::EDGetTokenT<reco::BeamSpot> token_bs;
    edm::EDGetTokenT<MeasurementTrackerEvent> token_measTrkEvt;
  };

  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer RecHitContainer;

  ElectronSeedGenerator(const edm::ParameterSet&, const Tokens&);

  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&,
           const edm::EventSetup& setup,
           const reco::SuperClusterRefVector&,
           const std::vector<float>& hoe1s,
           const std::vector<float>& hoe2s,
           const std::vector<const TrajectorySeedCollection*>& seedsV,
           reco::ElectronSeedCollection&);

private:
  void seedsFromThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster,
                            float hoe1,
                            float hoe2,
                            reco::ElectronSeedCollection& out,
                            const TrackerTopology* tTopo);
  void seedsFromRecHits(std::vector<std::pair<RecHitWithDist, ConstRecHitPointer> >& elePixelHits,
                        PropagationDirection& dir,
                        const GlobalPoint& vertexPos,
                        const reco::ElectronSeed::CaloClusterRef& cluster,
                        reco::ElectronSeedCollection& out,
                        bool positron);
  void seedsFromTrajectorySeeds(const std::vector<SeedWithInfo>& elePixelSeeds,
                                const reco::ElectronSeed::CaloClusterRef& cluster,
                                float hoe1,
                                float hoe2,
                                reco::ElectronSeedCollection& out,
                                bool positron);
  void addSeed(reco::ElectronSeed& seed, const SeedWithInfo* info, bool positron, reco::ElectronSeedCollection& out);
  bool prepareElTrackSeed(ConstRecHitPointer outerhit, ConstRecHitPointer innerhit, const GlobalPoint& vertexPos);

  const bool dynamicphiroad_;
  const bool fromTrackerSeeds_;
  edm::Handle<std::vector<reco::Vertex> > vertices_;
  const edm::EDGetTokenT<std::vector<reco::Vertex> > verticesTag_;

  edm::Handle<reco::BeamSpot> beamSpot_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;

  const float lowPtThreshold_;
  const float highPtThreshold_;
  const float nSigmasDeltaZ1_;     // first z window size if not using the reco vertex
  const float deltaZ1WithVertex_;  // first z window size when using the reco vertex
  const float sizeWindowENeg_;

  const float deltaPhi1Low_;
  const float deltaPhi1High_;

  // so that deltaPhi1 = deltaPhi1Coef1_ + deltaPhi1Coef2_/clusterEnergyT
  const double deltaPhi1Coef1_;
  const double deltaPhi1Coef2_;

  const std::vector<const TrajectorySeedCollection*>* initialSeedCollectionVector_ = nullptr;

  edm::ESHandle<MagneticField> magField_;
  edm::ESHandle<TrackerGeometry> trackerGeometry_;
  KFUpdator updator_;
  std::unique_ptr<PropagatorWithMaterial> propagator_;

  const MeasurementTracker* measurementTracker_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerEventTag_;

  const NavigationSchool* navigationSchool_;

  const edm::EventSetup* setup_;

  PRecHitContainer recHits_;
  PTrajectoryStateOnDet pts_;

  // keep cacheIds to get records only when necessary
  unsigned long long cacheIDMagField_;
  unsigned long long cacheIDNavSchool_;
  unsigned long long cacheIDCkfComp_;
  unsigned long long cacheIDTrkGeom_;

  const std::string measurementTrackerName_;

  const bool useRecoVertex_;

  const float deltaPhi2B_;
  const float deltaPhi2F_;

  const float phiMin2B_;
  const float phiMin2F_;
  const float phiMax2B_;
  const float phiMax2F_;

  PixelHitMatcher electronMatcher_;
  PixelHitMatcher positronMatcher_;
};

#endif  // ElectronSeedGenerator_H
