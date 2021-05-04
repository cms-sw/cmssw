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
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class ElectronSeedGenerator {
public:
  struct Tokens {
    edm::EDGetTokenT<std::vector<reco::Vertex> > token_vtx;
    edm::EDGetTokenT<reco::BeamSpot> token_bs;
  };

  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer RecHitContainer;

  ElectronSeedGenerator(const edm::ParameterSet&, const Tokens&, edm::ConsumesCollector&&);

  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&,
           const reco::SuperClusterRefVector&,
           const std::vector<const TrajectorySeedCollection*>& seedsV,
           reco::ElectronSeedCollection&);

private:
  void seedsFromThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster,
                            reco::BeamSpot const& beamSpot,
                            std::vector<reco::Vertex> const* vertices,
                            reco::ElectronSeedCollection& out);

  const bool dynamicPhiRoad_;

  const edm::EDGetTokenT<std::vector<reco::Vertex> > verticesTag_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  edm::ESWatcher<IdealMagneticFieldRecord> magneticFieldWatcher_;
  edm::ESWatcher<TrackerDigiGeometryRecord> trackerGeometryWatcher_;

  const float lowPtThresh_;
  const float highPtThresh_;
  const float nSigmasDeltaZ1_;     // first z window size if not using the reco vertex
  const float deltaZ1WithVertex_;  // first z window size when using the reco vertex
  const float sizeWindowENeg_;

  const float deltaPhi1Low_;
  const float deltaPhi1High_;

  // so that deltaPhi1 = dPhi1Coef1_ + dPhi1Coef2_/clusterEnergyT
  const double dPhi1Coef2_;
  const double dPhi1Coef1_;

  const std::vector<const TrajectorySeedCollection*>* initialSeedCollectionVector_ = nullptr;

  const bool useRecoVertex_;

  const float deltaPhi2B_;
  const float deltaPhi2F_;

  const float phiMin2B_;
  const float phiMin2F_;
  const float phiMax2B_;
  const float phiMax2F_;

  PixelHitMatcher matcher_;
};

#endif  // ElectronSeedGenerator_H
