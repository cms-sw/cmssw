#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "CommonTools/RecoAlgos/interface/KDTreeLinkerAlgo.h"

#include "TMath.h"

namespace rechit = edm::soa::col::pf::rechit;
namespace cluster = edm::soa::col::pf::cluster;

// This class is used to find all links between PreShower clusters and ECAL clusters
// using a KDTree algorithm.
// It is used in PFBlockAlgo.cc in the function links().
class KDTreeLinkerPSEcal : public KDTreeLinkerBase {
public:
  KDTreeLinkerPSEcal(const edm::ParameterSet &conf);
  ~KDTreeLinkerPSEcal() override;

  // The KDTree building from rechits list.
  void buildTree(const PFTables &pftables) override;

  // Here we will iterate over all psCluster. For each one, we will search the closest
  // rechits in the KDTree, from rechits we will find the ecalClusters and after that
  // we will check the links between the psCluster and all closest ecalClusters.
  void searchLinks(const PFTables &pftables, reco::PFMultiLinksIndex &multilinks) override;

  // Here we free all allocated structures.
  void clear() override;

private:
  // This method allows us to build the "tree" from the "rechitsSet".
  void buildTree(const PFTables &pftables, const std::vector<size_t> &rechitsSet, KDTreeLinkerAlgo<size_t> &tree);

private:
  // Some const values.
  const double resPSpitch_;
  const double resPSlength_;
  const double ps1ToEcal_;  // ratio : zEcal / zPS1
  const double ps2ToEcal_;  // ration : zEcal / zPS2

  // KD trees
  KDTreeLinkerAlgo<size_t> treeNeg_;
  KDTreeLinkerAlgo<size_t> treePos_;
};

// the text name is different so that we can easily
// construct it when calling the factory
DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, KDTreeLinkerPSEcal, "KDTreePreshowerAndECALLinker");

KDTreeLinkerPSEcal::KDTreeLinkerPSEcal(const edm::ParameterSet &conf)
    : KDTreeLinkerBase(conf), resPSpitch_(0.19), resPSlength_(6.1), ps1ToEcal_(1.072), ps2ToEcal_(1.057) {}

KDTreeLinkerPSEcal::~KDTreeLinkerPSEcal() { clear(); }

void KDTreeLinkerPSEcal::buildTree(const PFTables &pftables) {
  std::vector<size_t> rechits_neg;
  std::vector<size_t> rechits_pos;
  const auto &clusters = pftables.getClusterTable(_fieldType);
  for (size_t icluster = 0; icluster < clusters.cluster_table.size(); icluster++) {
    const auto layer = clusters.cluster_table.get<cluster::Layer>(icluster);
    if (layer == PFLayer::ECAL_ENDCAP) {
      const auto posz = clusters.cluster_table.get<cluster::Posz>(icluster);
      for (size_t irechit : clusters.cluster_to_rechit.at(icluster)) {
        if (posz < 0) {
          rechits_neg.push_back(irechit);
        } else {
          rechits_pos.push_back(irechit);
        }
      }
    }
  }

  buildTree(pftables, rechits_neg, treeNeg_);
  buildTree(pftables, rechits_pos, treePos_);
}

void KDTreeLinkerPSEcal::buildTree(const PFTables &pftables,
                                   const std::vector<size_t> &rechitsSet,
                                   KDTreeLinkerAlgo<size_t> &tree) {
  // List of pseudo-rechits that will be used to create the KDTree
  std::vector<KDTreeNodeInfo<size_t, 2>> eltList;

  const auto &clusters = pftables.getClusterTable(_fieldType);

  // Filling of this eltList
  for (size_t irechit : rechitsSet) {
    const float x = clusters.rechit_table.get<rechit::Posx>(irechit);
    const float y = clusters.rechit_table.get<rechit::Posy>(irechit);

    KDTreeNodeInfo<size_t, 2> rhinfo{irechit, x, y};
    eltList.push_back(rhinfo);
  }

  // xmin-xmax, ymain-ymax
  KDTreeBox region{-150.f, 150.f, -150.f, 150.f};

  // We may now build the KDTree
  tree.build(eltList, region);
}

void KDTreeLinkerPSEcal::searchLinks(const PFTables &pftables, reco::PFMultiLinksIndex &multilinks) {
  // Most of the code has been taken from LinkByRecHit.cc
  LogDebug("KDTreeLinkerPSEcal") << "searchLinks _fieldType=" << _fieldType << " targetType=" << _targetType;

  const auto &clustersPS = pftables.getClusterTable(_targetType);
  const auto &clustersECAL = pftables.getClusterTable(_fieldType);
  const auto &ecal_rh = clustersECAL.rechit_table;

  // We iterate over the PS clusters.
  for (size_t ips = 0; ips < clustersPS.cluster_table.size(); ips++) {
    // PS cluster position, extrapolated to ECAL
    double zPS = clustersPS.cluster_table.get<cluster::Posz>(ips);
    double xPS = clustersPS.cluster_table.get<cluster::Posx>(ips);
    double yPS = clustersPS.cluster_table.get<cluster::Posy>(ips);

    double etaPS = std::abs(clustersPS.cluster_table.get<cluster::Eta>(ips));
    double deltaX = 0.;
    double deltaY = 0.;
    float xPSonEcal = xPS;
    float yPSonEcal = yPS;

    if (clustersPS.cluster_table.get<cluster::Layer>(ips) == PFLayer::PS1) {  // PS1

      // vertical strips, measure x with pitch precision
      deltaX = resPSpitch_;
      deltaY = resPSlength_;
      xPSonEcal *= ps1ToEcal_;
      yPSonEcal *= ps1ToEcal_;

    } else {  // PS2

      // horizontal strips, measure y with pitch precision
      deltaY = resPSpitch_;
      deltaX = resPSlength_;
      xPSonEcal *= ps2ToEcal_;
      yPSonEcal *= ps2ToEcal_;
    }

    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.

    double maxEcalRadius = cristalXYMaxSize_ / 2.;

    // The inflation factor includes the approximate projection from Preshower to ECAL
    double inflation = 2.4 - (etaPS - 1.6);
    float rangeX = maxEcalRadius * (1 + (0.05 + 1.0 / maxEcalRadius * deltaX / 2.)) * inflation;
    float rangeY = maxEcalRadius * (1 + (0.05 + 1.0 / maxEcalRadius * deltaY / 2.)) * inflation;

    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<size_t> recHits;
    KDTreeBox trackBox(xPSonEcal - rangeX, xPSonEcal + rangeX, yPSonEcal - rangeY, yPSonEcal + rangeY);

    if (zPS < 0)
      treeNeg_.search(trackBox, recHits);
    else
      treePos_.search(trackBox, recHits);

    for (size_t irecHit : recHits) {
      // Find all clusters associated to given rechit
      const auto &rechit_clusters = clustersECAL.rechit_to_cluster.at(irecHit);

      for (size_t icluster : rechit_clusters) {
        double clusterz = clustersECAL.cluster_table.get<cluster::Posz>(icluster);
        const auto zPS_over_cluster = zPS / clusterz;

        auto rechit_corner_posx = ecal_rh.get<rechit::CornerXBV>(irecHit);
        auto rechit_corner_posy = ecal_rh.get<rechit::CornerYBV>(irecHit);
        for (size_t icorner = 0; icorner < 4; icorner++) {
          rechit_corner_posx[icorner] *= zPS_over_cluster;
          rechit_corner_posy[icorner] *= zPS_over_cluster;
        }
        const double rechit_posx = ecal_rh.get<rechit::Posx>(irecHit) * zPS_over_cluster;
        const double rechit_posy = ecal_rh.get<rechit::Posy>(irecHit) * zPS_over_cluster;

        double x[5];
        double y[5];
        for (unsigned jc = 0; jc < 4; ++jc) {
          x[3 - jc] = rechit_corner_posx[jc] +
                      (rechit_corner_posx[jc] - rechit_posx) *
                          (0.05 + 1.0 / std::abs((rechit_corner_posx[jc] - rechit_posx)) * deltaX / 2.);
          y[3 - jc] = rechit_corner_posy[jc] +
                      (rechit_corner_posy[jc] - rechit_posy) *
                          (0.05 + 1.0 / std::abs((rechit_corner_posy[jc] - rechit_posy)) * deltaY / 2.);
        }

        x[4] = x[0];
        y[4] = y[0];

        bool isinside = TMath::IsInside(xPS, yPS, 5, x, y);

        // Check if the track and the cluster are linked
        if (isinside) {
          multilinks.addLink(
              clustersPS.cluster_to_element[ips], clustersECAL.cluster_to_element[icluster], _targetType, _fieldType);
          LogTrace("KDTreeLinkerPSEcal") << " ips_elem=" << clustersPS.cluster_to_element[ips]
                                         << " icluster_elem=" << clustersECAL.cluster_to_element[icluster];
        }
      }
    }
  }
}

void KDTreeLinkerPSEcal::clear() {
  treeNeg_.clear();
  treePos_.clear();
}
