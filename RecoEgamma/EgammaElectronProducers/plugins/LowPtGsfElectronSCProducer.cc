#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <iostream>

class LowPtGsfElectronSCProducer : public edm::stream::EDProducer<> {
public:
  explicit LowPtGsfElectronSCProducer(const edm::ParameterSet&);

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  reco::PFClusterRef closestCluster(const reco::PFTrajectoryPoint& point,
                                    const edm::Handle<reco::PFClusterCollection>& clusters,
                                    std::vector<int>& matched);

  const edm::EDGetTokenT<reco::GsfPFRecTrackCollection> gsfPfRecTracks_;
  const edm::EDGetTokenT<reco::PFClusterCollection> ecalClusters_;
  const double dr2_;
};

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSCProducer::LowPtGsfElectronSCProducer(const edm::ParameterSet& cfg)
    : gsfPfRecTracks_{consumes<reco::GsfPFRecTrackCollection>(cfg.getParameter<edm::InputTag>("gsfPfRecTracks"))},
      ecalClusters_{consumes<reco::PFClusterCollection>(cfg.getParameter<edm::InputTag>("ecalClusters"))},
      dr2_{cfg.getParameter<double>("MaxDeltaR2")} {
  produces<reco::CaloClusterCollection>();
  produces<reco::SuperClusterCollection>();
  produces<edm::ValueMap<reco::SuperClusterRef> >();
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSCProducer::produce(edm::Event& event, const edm::EventSetup&) {
  // Input GsfPFRecTracks collection
  auto gsfPfRecTracks = edm::makeValid(event.getHandle(gsfPfRecTracks_));

  // Input EcalClusters collection
  reco::PFClusterCollection const& ecalClusters = event.get(ecalClusters_);

  // Output SuperClusters collection and getRefBeforePut
  auto superClusters = std::make_unique<reco::SuperClusterCollection>(reco::SuperClusterCollection());
  superClusters->reserve(gsfPfRecTracks->size());
  const reco::SuperClusterRefProd superClustersRefProd = event.getRefBeforePut<reco::SuperClusterCollection>();

  // Output ValueMap container of GsfPFRecTrackRef index to SuperClusterRef
  std::vector<reco::SuperClusterRef> superClustersValueMap;

  // Output CaloClusters collection
  auto caloClusters = std::make_unique<reco::CaloClusterCollection>(reco::CaloClusterCollection());
  caloClusters->reserve(ecalClusters.size());

  // Index[GSF track][trajectory point] for "best" CaloCluster
  std::vector<std::vector<int> > cluster_idx;
  cluster_idx.reserve(gsfPfRecTracks->size());

  // Index[GSF track][trajectory point] for "best" PFCluster
  std::vector<std::vector<int> > pfcluster_idx;
  pfcluster_idx.reserve(gsfPfRecTracks->size());

  // dr2min[GSF track][trajectory point] for "best" CaloCluster
  std::vector<std::vector<float> > cluster_dr2min;
  cluster_dr2min.reserve(gsfPfRecTracks->size());

  // Construct list of trajectory points from the GSF track and electron brems
  std::vector<std::vector<const reco::PFTrajectoryPoint*> > points;
  points.reserve(gsfPfRecTracks->size());
  for (auto const& trk : *gsfPfRecTracks) {
    // Extrapolated track
    std::vector<const reco::PFTrajectoryPoint*> traj;
    traj.reserve(trk.PFRecBrem().size() + 1);
    traj.push_back(&trk.extrapolatedPoint(reco::PFTrajectoryPoint::LayerType::ECALShowerMax));
    // Extrapolated brem trajectories
    for (auto const& brem : trk.PFRecBrem()) {
      traj.push_back(&brem.extrapolatedPoint(reco::PFTrajectoryPoint::LayerType::ECALShowerMax));
    }
    auto size = traj.size();
    points.push_back(std::move(traj));
    // Size containers
    cluster_idx.emplace_back(size, -1);
    pfcluster_idx.emplace_back(size, -1);
    cluster_dr2min.emplace_back(size, 1.e6);
  }

  // For each cluster, find closest trajectory point ("global" arbitration at event level)
  for (size_t iclu = 0; iclu < ecalClusters.size(); ++iclu) {  // Cluster loop
    std::pair<int, int> point = std::make_pair(-1, -1);
    float dr2min = 1.e6;
    for (size_t ipoint = 0; ipoint < points.size(); ++ipoint) {            // GSF track loop
      for (size_t jpoint = 0; jpoint < points[ipoint].size(); ++jpoint) {  // Traj point loop
        if (points[ipoint][jpoint]->isValid()) {
          float dr2 = reco::deltaR2(ecalClusters[iclu], points[ipoint][jpoint]->positionREP());
          if (dr2 < dr2min) {
            // Store nearest point to this cluster
            dr2min = dr2;
            point = std::make_pair(ipoint, jpoint);
          }
        }
      }
    }
    if (point.first >= 0 && point.second >= 0 &&               // if this cluster is matched to a point ...
        dr2min < cluster_dr2min[point.first][point.second]) {  // ... and cluster is closest to the same point
      // Copy CaloCluster to new collection
      caloClusters->push_back(ecalClusters[iclu]);
      // Store cluster index for creation of SC later
      cluster_idx[point.first][point.second] = caloClusters->size() - 1;
      pfcluster_idx[point.first][point.second] = iclu;
      cluster_dr2min[point.first][point.second] = dr2min;
    }
  }

  // Put CaloClusters in event and get orphan handle
  const edm::OrphanHandle<reco::CaloClusterCollection>& caloClustersH = event.put(std::move(caloClusters));

  // Loop through GSF tracks
  for (size_t itrk = 0; itrk < gsfPfRecTracks->size(); ++itrk) {
    // Used to create SC
    float energy = 0.;
    float X = 0., Y = 0., Z = 0.;
    reco::CaloClusterPtr seed;
    reco::CaloClusterPtrVector clusters;
    std::vector<const reco::PFCluster*> barePtrs;

    // Find closest match in dr2 from points associated to given track
    int index = -1;
    float dr2 = 1.e6;
    for (size_t ipoint = 0; ipoint < cluster_idx[itrk].size(); ++ipoint) {
      if (cluster_idx[itrk][ipoint] < 0) {
        continue;
      }
      if (cluster_dr2min[itrk][ipoint] < dr2) {
        dr2 = cluster_dr2min[itrk][ipoint];
        index = cluster_idx[itrk][ipoint];
      }
    }

    // For each track, loop through points and use associated cluster
    for (size_t ipoint = 0; ipoint < cluster_idx[itrk].size(); ++ipoint) {
      if (cluster_idx[itrk][ipoint] < 0) {
        continue;
      }
      reco::CaloClusterPtr clu(caloClustersH, cluster_idx[itrk][ipoint]);
      if (clu.isNull()) {
        continue;
      }
      if (!(cluster_dr2min[itrk][ipoint] < dr2_ ||  // Require cluster to be closer than dr2_ ...
            index == cluster_idx[itrk][ipoint])) {
        continue;
      }  // ... unless it is the closest one ...
      if (seed.isNull()) {
        seed = clu;
      }
      clusters.push_back(clu);
      energy += clu->correctedEnergy();
      X += clu->position().X() * clu->correctedEnergy();
      Y += clu->position().Y() * clu->correctedEnergy();
      Z += clu->position().Z() * clu->correctedEnergy();
      auto index = pfcluster_idx[itrk][ipoint];
      if (index < static_cast<decltype(index)>(ecalClusters.size())) {
        barePtrs.push_back(&(ecalClusters[index]));
      }
    }
    X /= energy;
    Y /= energy;
    Z /= energy;

    // Create SC
    if (seed.isNonnull()) {
      reco::SuperCluster sc(energy, math::XYZPoint(X, Y, Z));
      sc.setCorrectedEnergy(energy);
      sc.setSeed(seed);
      sc.setClusters(clusters);
      PFClusterWidthAlgo pfwidth(barePtrs);
      sc.setEtaWidth(pfwidth.pflowEtaWidth());
      sc.setPhiWidth(pfwidth.pflowPhiWidth());
      sc.rawEnergy();  // Cache the value of raw energy
      superClusters->push_back(sc);

      // Populate ValueMap container
      superClustersValueMap.push_back(reco::SuperClusterRef(superClustersRefProd, superClusters->size() - 1));
    } else {
      superClustersValueMap.push_back(reco::SuperClusterRef(superClustersRefProd.id()));
    }

  }  // GSF tracks

  // Put SuperClusters in event
  event.put(std::move(superClusters));

  auto ptr = std::make_unique<edm::ValueMap<reco::SuperClusterRef> >(edm::ValueMap<reco::SuperClusterRef>());
  edm::ValueMap<reco::SuperClusterRef>::Filler filler(*ptr);
  filler.insert(gsfPfRecTracks, superClustersValueMap.begin(), superClustersValueMap.end());
  filler.fill();
  event.put(std::move(ptr));
}

////////////////////////////////////////////////////////////////////////////////
//
reco::PFClusterRef LowPtGsfElectronSCProducer::closestCluster(const reco::PFTrajectoryPoint& point,
                                                              const edm::Handle<reco::PFClusterCollection>& clusters,
                                                              std::vector<int>& matched) {
  reco::PFClusterRef closest;
  if (point.isValid()) {
    float dr2min = dr2_;
    for (size_t ii = 0; ii < clusters->size(); ++ii) {
      if (std::find(matched.begin(), matched.end(), ii) == matched.end()) {
        float dr2 = reco::deltaR2(clusters->at(ii), point.positionREP());
        if (dr2 < dr2min) {
          closest = reco::PFClusterRef(clusters, ii);
          dr2min = dr2;
        }
      }
    }
    if (dr2min < (dr2_ - 1.e-6)) {
      matched.push_back(closest.index());
    }
  }
  return closest;
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSCProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfPfRecTracks", edm::InputTag("lowPtGsfElePfGsfTracks"));
  desc.add<edm::InputTag>("ecalClusters", edm::InputTag("particleFlowClusterECAL"));
  desc.add<edm::InputTag>("hcalClusters", edm::InputTag("particleFlowClusterHCAL"));
  desc.add<double>("MaxDeltaR2", 0.5);
  descriptions.add("lowPtGsfElectronSuperClusters", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronSCProducer);
