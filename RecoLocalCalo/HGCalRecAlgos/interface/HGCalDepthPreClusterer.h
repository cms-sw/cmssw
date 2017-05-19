#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalDepthPreClusterer
#define RecoLocalCalo_HGCalRecAlgos_HGCalDepthPreClusterer


#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include <list>

#include "RecoLocalCalo/HGCalRecAlgos/interface/ClusterTools.h"

class HGCalDepthPreClusterer 
{
public:
  
 HGCalDepthPreClusterer() : radius(0.), minClusters(0.), clusterTools(nullptr)
    {
  }
  
 HGCalDepthPreClusterer(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes, double radius_in, uint32_t min_clusters) : 
  radius(radius_in),
  minClusters(min_clusters),
  clusterTools(std::make_unique<hgcal::ClusterTools>(conf,sumes)) {
  }

  void getEvent(const edm::Event& ev) { clusterTools->getEvent(ev); }
  void getEventSetup(const edm::EventSetup& es) { clusterTools->getEventSetup(es); }

  typedef std::vector<reco::BasicCluster> ClusterCollection;
  //  typedef std::vector<reco::BasicCluster> MultiCluster;

  std::vector<reco::HGCalMultiCluster> makePreClusters(const reco::HGCalMultiCluster::ClusterCollection &) const;    

private:  
  float radius;
  uint32_t minClusters;
  
  std::unique_ptr<hgcal::ClusterTools> clusterTools;

};

#endif
