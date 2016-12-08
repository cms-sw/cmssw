#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalDepthPreClusterer
#define RecoLocalCalo_HGCalRecAlgos_HGCalDepthPreClusterer


#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include <list>



class HGCalDepthPreClusterer 
{
public:
  
  HGCalDepthPreClusterer() : radius(0.) {
  }
  
  HGCalDepthPreClusterer(double radius_in) : radius(radius_in) {
  }

  typedef std::vector<reco::BasicCluster> ClusterCollection;
  //  typedef std::vector<reco::BasicCluster> MultiCluster;

  std::vector<reco::HGCalMultiCluster> makePreClusters(const reco::HGCalMultiCluster::ClusterCollection &) const;    

private:  
  float radius;
};

#endif
