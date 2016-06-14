#ifndef RecoHGCAL_HGCALClusters_HGCalDepthPreClusterer
#define RecoHGCAL_HGCALClusters_HGCalDepthPreClusterer


#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include <list>

class HGCalMultiCluster;

class HGCalDepthPreClusterer 
{
public:
  
  HGCalDepthPreClusterer() : radius(0.)
  {
  }
  
  HGCalDepthPreClusterer(double radius_in) : radius(radius_in)
  {
  }

  typedef std::vector<reco::BasicCluster> ClusterCollection;
  //  typedef std::vector<reco::BasicCluster> MultiCluster;

  std::vector<HGCalMultiCluster> makePreClusters(const ClusterCollection &);    

private:
  float dist(const reco::BasicCluster &, const reco::BasicCluster &);

  float radius;
  std::vector<HGCalMultiCluster> thePreClusters;

};

#endif
