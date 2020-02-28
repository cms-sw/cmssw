#ifndef RecoLocalCalo_HGCalRecAlgos_HGCal3DClustering
#define RecoLocalCalo_HGCalRecAlgos_HGCal3DClustering

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include <vector>
#include <array>

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/ClusterTools.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

#include "CommonTools/RecoAlgos/interface/KDTreeLinkerAlgo.h"

class HGCal3DClustering {
public:
  HGCal3DClustering() : radii({0., 0., 0.}), minClusters(0), clusterTools(nullptr) {}

  HGCal3DClustering(const edm::ParameterSet& conf,
                    edm::ConsumesCollector& sumes,
                    const std::vector<double>& radii_in,
                    uint32_t min_clusters)
      : radii(radii_in),
        minClusters(min_clusters),
        es(0),
        clusterTools(std::make_unique<hgcal::ClusterTools>(conf, sumes)) {}

  HGCal3DClustering(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
      : HGCal3DClustering(conf,
                          sumes,
                          conf.getParameter<std::vector<double>>("multiclusterRadii"),
                          conf.getParameter<unsigned>("minClusters")) {}

  void getEvent(const edm::Event& ev) { clusterTools->getEvent(ev); }
  void getEventSetup(const edm::EventSetup& es) {
    clusterTools->getEventSetup(es);
    rhtools_.getEventSetup(es);
    maxlayer = rhtools_.lastLayerBH();
    points.clear();
    minpos.clear();
    maxpos.clear();
    zees.clear();
    points.resize(2 * (maxlayer + 1));
    minpos.resize(2 * (maxlayer + 1), {{0.0f, 0.0f}});
    maxpos.resize(2 * (maxlayer + 1), {{0.0f, 0.0f}});
    zees.resize(2 * (maxlayer + 1), 0.);
  }

  typedef std::vector<reco::BasicCluster> ClusterCollection;

  std::vector<reco::HGCalMultiCluster> makeClusters(const reco::HGCalMultiCluster::ClusterCollection&);

private:
  void organizeByLayer(const reco::HGCalMultiCluster::ClusterCollection&);
  void reset() {
    for (auto& it : points) {
      it.clear();
      std::vector<KDNode>().swap(it);
    }
    std::fill(zees.begin(), zees.end(), 0.);
    for (unsigned int i = 0; i < minpos.size(); i++) {
      minpos[i][0] = 0.;
      minpos[i][1] = 0.;
      maxpos[i][0] = 0.;
      maxpos[i][1] = 0.;
    }
  }
  void layerIntersection(std::array<double, 3>& to, const std::array<double, 3>& from) const;

  //max number of layers
  unsigned int maxlayer;

  std::vector<double> radii;
  uint32_t minClusters;
  struct ClusterRef {
    int ind;
    float z;
    ClusterRef(int ind_i, float z_i) : ind(ind_i), z(z_i) {}
    ClusterRef() : ind(-1), z(0.) {}
  };

  typedef KDTreeLinkerAlgo<ClusterRef> KDTree;
  typedef KDTreeNodeInfo<ClusterRef> KDNode;
  std::vector<std::vector<KDNode>> points;
  std::vector<std::array<float, 2>> minpos;
  std::vector<std::array<float, 2>> maxpos;
  std::vector<size_t> es;                            /*!< vector to contain sorted indices of all clusters. */
  std::vector<float> zees;                           /*!< vector to contain z position of each layer. */
  std::unique_ptr<hgcal::ClusterTools> clusterTools; /*!< instance of tools to simplify cluster access. */
  hgcal::RecHitTools rhtools_;                       /*!< instance of tools to access RecHit information. */
};

#endif
