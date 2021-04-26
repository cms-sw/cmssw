#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalDepthPreClusterer
#define RecoLocalCalo_HGCalRecAlgos_HGCalDepthPreClusterer

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include <list>

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/ClusterTools.h"

class HGCalDepthPreClusterer {
public:
  HGCalDepthPreClusterer()
      : radii({
            0.,
            0.,
            0.,
        }),
        minClusters(0),
        realSpaceCone(false),
        clusterTools(nullptr) {}

  HGCalDepthPreClusterer(const edm::ParameterSet& conf,
                         edm::ConsumesCollector& sumes,
                         std::vector<float> radii_in,
                         uint32_t min_clusters,
                         bool real_space_cone)
      : radii(radii_in),
        minClusters(min_clusters),
        realSpaceCone(real_space_cone),
        clusterTools(std::make_unique<hgcal::ClusterTools>(conf, sumes)),
        caloGeomToken_(sumes.esConsumes<CaloGeometry, CaloGeometryRecord>()) {}

  void getEvent(const edm::Event& ev) { clusterTools->getEvent(ev); }
  void getEventSetup(const edm::EventSetup& es) {
    clusterTools->getEventSetup(es);
    edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeomToken_);
    rhtools_.setGeometry(*geom);
  }

  typedef std::vector<reco::BasicCluster> ClusterCollection;
  //  typedef std::vector<reco::BasicCluster> MultiCluster;

  std::vector<reco::HGCalMultiCluster> makePreClusters(const reco::HGCalMultiCluster::ClusterCollection&) const;

private:
  std::vector<float> radii;
  uint32_t minClusters;
  bool realSpaceCone; /*!< flag to use cartesian space clustering. */

  std::unique_ptr<hgcal::ClusterTools> clusterTools;
  hgcal::RecHitTools rhtools_; /*!< instance of tools to access RecHit information. */
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
};

#endif
