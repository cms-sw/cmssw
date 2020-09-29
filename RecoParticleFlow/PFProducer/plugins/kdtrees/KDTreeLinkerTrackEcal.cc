#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "CommonTools/RecoAlgos/interface/KDTreeLinkerAlgo.h"
#include "FWCore/SOA/interface/Column.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"

using namespace edm::soa;
using namespace edm::soa::col;

// This class is used to find all links between Tracks and ECAL clusters
// using a KDTree algorithm.
// It is used in PFBlockAlgo.cc in the function links().
class KDTreeLinkerTrackEcal : public KDTreeLinkerBase {
public:
  KDTreeLinkerTrackEcal(const edm::ParameterSet& conf);
  ~KDTreeLinkerTrackEcal() override;

  // The KDTree building from rechits list.
  void buildTree(const PFTables& pftables) override;

  // Here we will iterate over all tracks. For each track intersection point with ECAL,
  // we will search the closest rechits in the KDTree, from rechits we will find the
  // ecalClusters and after that we will check the links between the track and
  // all closest ecalClusters.
  void searchLinks(const PFTables& pftables, reco::PFMultiLinksIndex& multilinks) override;

  // Here we free all allocated structures.
  void clear() override;

private:
  // KD trees
  KDTreeLinkerAlgo<size_t> tree_;
};

// the text name is different so that we can easily
// construct it when calling the factory
DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, KDTreeLinkerTrackEcal, "KDTreeTrackAndECALLinker");

KDTreeLinkerTrackEcal::KDTreeLinkerTrackEcal(const edm::ParameterSet& conf) : KDTreeLinkerBase(conf) {}

KDTreeLinkerTrackEcal::~KDTreeLinkerTrackEcal() { clear(); }

void KDTreeLinkerTrackEcal::buildTree(const PFTables& pftables) {
  const auto& rechitTable = pftables.clusters_ecal_.rechit_table_;

  // List of pseudo-rechits that will be used to create the KDTree
  std::vector<KDTreeNodeInfo<size_t, 2>> eltList;

  // Filling of this list
  for (size_t irechit = 0; irechit < rechitTable.size(); irechit++) {
    KDTreeNodeInfo<size_t, 2> rh1(
        irechit, float(rechitTable.get<pf::rechit::Eta>(irechit)), float(rechitTable.get<pf::rechit::Phi>(irechit)));
    eltList.push_back(rh1);

    // Here we solve the problem of phi circular set by duplicating some rechits
    // too close to -Pi (or to Pi) and adding (substracting) to them 2 * Pi.
    if (rh1.dims[1] > (M_PI - phiOffset_)) {
      float phi = rh1.dims[1] - 2 * M_PI;
      KDTreeNodeInfo<size_t, 2> rh2(irechit, float(rechitTable.get<pf::rechit::Eta>(irechit)), phi);
      eltList.push_back(rh2);
    }

    if (rh1.dims[1] < (M_PI * -1.0 + phiOffset_)) {
      float phi = rh1.dims[1] + 2 * M_PI;
      KDTreeNodeInfo<size_t, 2> rh3(irechit, float(rechitTable.get<pf::rechit::Eta>(irechit)), phi);
      eltList.push_back(rh3);
    }
  }

  // Here we define the upper/lower bounds of the 2D space (eta/phi).
  float phimin = -1.0 * M_PI - phiOffset_;
  float phimax = M_PI + phiOffset_;

  // etamin-etamax, phimin-phimax
  KDTreeBox region(-3.0f, 3.0f, phimin, phimax);

  // We may now build the KDTree
  tree_.build(eltList, region);
}

void KDTreeLinkerTrackEcal::searchLinks(const PFTables& pftables, reco::PFMultiLinksIndex& multilinks) {
  // Most of the code has been taken from LinkByRecHit.cc
  LogDebug("KDTreeLinkerTrackEcal") << "searchLinks fieldType=" << _fieldType << " _targetType=" << _targetType;

  const auto& trackTableVtx_ = pftables.track_table_vertex_;
  const auto& trackTable_ = pftables.track_table_ecalshowermax_;
  const auto& clusterTable_ = pftables.clusters_ecal_.cluster_table_;
  const auto& rechitTable = pftables.clusters_ecal_.rechit_table_;

  // We iterate over the tracks.
  for (size_t itrack = 0; itrack < trackTableVtx_.size(); itrack++) {
    if (not trackTable_.get<pf::track::ExtrapolationValid>(itrack)) {
      continue;
    }

    const auto trackPt = trackTableVtx_.get<col::pf::track::Pt>(itrack);
    const float tracketa = trackTable_.get<col::pf::track::Eta>(itrack);
    const float trackphi = trackTable_.get<col::pf::track::Phi>(itrack);
    const auto trackx = trackTable_.get<col::pf::track::Posx>(itrack);
    const auto tracky = trackTable_.get<col::pf::track::Posy>(itrack);
    const auto trackz = trackTable_.get<col::pf::track::Posz>(itrack);

    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    float range = cristalPhiEtaMaxSize_ * (2.0 + 1.0 / std::min(1., trackPt / 2.));

    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<size_t> recHits;
    KDTreeBox trackBox(tracketa - range, tracketa + range, trackphi - range, trackphi + range);
    tree_.search(trackBox, recHits);

    // Here we check all rechit candidates using the non-approximated method.
    for (size_t irecHit : recHits) {
      double rhsizeeta = std::abs(rechitTable.get<pf::rechit::CornerEta>(irecHit)[3] -
                                  rechitTable.get<pf::rechit::CornerEta>(irecHit)[1]);
      double rhsizephi = std::abs(rechitTable.get<pf::rechit::CornerPhi>(irecHit)[3] -
                                  rechitTable.get<pf::rechit::CornerPhi>(irecHit)[1]);
      if (rhsizephi > M_PI)
        rhsizephi = 2. * M_PI - rhsizephi;

      double deta = std::abs(rechitTable.get<pf::rechit::Eta>(irecHit) - tracketa);
      double dphi = std::abs(rechitTable.get<pf::rechit::Phi>(irecHit) - trackphi);
      if (dphi > M_PI)
        dphi = 2. * M_PI - dphi;

      LogTrace("KDTreeLinkerTrackEcal") << "getting rechit " << irecHit;

      // Find all clusters associated to given rechit
      const auto& rechit_clusters = pftables.clusters_ecal_.rechit_to_cluster_.at(irecHit);

      for (const size_t clusteridx : rechit_clusters) {
        double clusterz = clusterTable_.get<pf::cluster::Posz>(clusteridx);
        int fracsNbr = clusterTable_.get<pf::cluster::FracsNbr>(clusteridx);

        if (clusterTable_.get<pf::cluster::Layer>(clusteridx) == PFLayer::ECAL_BARREL) {  // BARREL
          // Check if the track is in the barrel
          if (std::abs(trackz) > 300.)
            continue;

          double _rhsizeeta = rhsizeeta * (2.00 + 1.0 / (fracsNbr * std::min(1., trackPt / 2.)));
          double _rhsizephi = rhsizephi * (2.00 + 1.0 / (fracsNbr * std::min(1., trackPt / 2.)));

          // Check if the track and the cluster are linked
          if (deta < (_rhsizeeta / 2.) && dphi < (_rhsizephi / 2.)) {
            multilinks.addLink(pftables.clusters_ecal_.cluster_to_element_[clusteridx],
                               pftables.track_to_element_[itrack],
                               _fieldType,
                               _targetType);
            LogTrace("KDTreeLinkerTrackEcal")
                << "itrack=" << itrack << " tracketa=" << tracketa << " trackphi=" << trackphi
                << " icluster=" << clusteridx << " rheta=" << rechitTable.get<pf::rechit::Eta>(irecHit)
                << " rhphi=" << rechitTable.get<pf::rechit::Phi>(irecHit)
                << " icluster_elem=" << pftables.clusters_ecal_.cluster_to_element_[clusteridx]
                << " itrack_elem=" << pftables.track_to_element_[itrack];
          }

        } else {  // ENDCAP

          // Check if the track is in the cap
          if (std::abs(trackz) < 300.)
            continue;
          if (trackz * clusterz < 0.)
            continue;

          double x[5];
          double y[5];
          const auto& rechit_corner_posx = rechitTable.get<pf::rechit::CornerX>(irecHit);
          const auto& rechit_corner_posy = rechitTable.get<pf::rechit::CornerY>(irecHit);

          for (unsigned jc = 0; jc < 4; ++jc) {
            x[3 - jc] = rechit_corner_posx[jc] + (rechit_corner_posx[jc] - rechitTable.get<pf::rechit::Posx>(irecHit)) *
                                                     (1.00 + 0.50 / fracsNbr / std::min(1., trackPt / 2.));
            y[3 - jc] = rechit_corner_posy[jc] + (rechit_corner_posy[jc] - rechitTable.get<pf::rechit::Posy>(irecHit)) *
                                                     (1.00 + 0.50 / fracsNbr / std::min(1., trackPt / 2.));
          }

          x[4] = x[0];
          y[4] = y[0];

          bool isinside = TMath::IsInside(trackx, tracky, 5, x, y);

          // Check if the track and the cluster are linked
          if (isinside) {
            multilinks.addLink(pftables.clusters_ecal_.cluster_to_element_[clusteridx],
                               pftables.track_to_element_[itrack],
                               _fieldType,
                               _targetType);
            LogTrace("KDTreeLinkerTrackEcal")
                << "itrack=" << itrack << " tracketa=" << tracketa << " trackphi=" << trackphi
                << " icluster=" << clusteridx << " rheta=" << rechitTable.get<pf::rechit::Eta>(irecHit)
                << " rhphi=" << rechitTable.get<pf::rechit::Phi>(irecHit)
                << " icluster_elem=" << pftables.clusters_ecal_.cluster_to_element_[clusteridx]
                << " itrack_elem=" << pftables.track_to_element_[itrack];
          }
        }  //cluster layer
      }    //loop over clusters of rechit
    }      // loop over rechits
  }        // loop over tracks
  LogDebug("KDTreeLinkerTrackEcal") << "searchLinks done";
}

void KDTreeLinkerTrackEcal::clear() { tree_.clear(); }
