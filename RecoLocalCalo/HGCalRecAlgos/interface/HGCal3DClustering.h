#ifndef RecoLocalCalo_HGCalRecAlgos_HGCal3DClustering
#define RecoLocalCalo_HGCalRecAlgos_HGCal3DClustering


#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include <list>

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/ClusterTools.h"

#include "KDTreeLinkerAlgoT.h"

class HGCal3DClustering 
{
public:
  
 HGCal3DClustering() : radius(0.), minClusters(0.), clusterTools(nullptr)
  {
  }
  
 HGCal3DClustering(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes, double radius_in, uint32_t min_clusters) : 
  radius(radius_in),
  minClusters(min_clusters),
  points(2*(maxlayer+1)),
  minpos(2*(maxlayer+1),{ {0.0f,0.0f} }),
  maxpos(2*(maxlayer+1),{ {0.0f,0.0f} }),
  es(0),
  zees(2*(maxlayer+1),0.),
  clusterTools(std::make_unique<hgcal::ClusterTools>(conf,sumes)) 
  {
  }

  void getEvent(const edm::Event& ev) { clusterTools->getEvent(ev); }
  void getEventSetup(const edm::EventSetup& es) 
  { 
    clusterTools->getEventSetup(es); 
    rhtools_.getEventSetup(es);
  }

  typedef std::vector<reco::BasicCluster> ClusterCollection;
  //  typedef std::vector<reco::BasicCluster> MultiCluster;

  std::vector<reco::HGCalMultiCluster> makeClusters(const reco::HGCalMultiCluster::ClusterCollection &);    

private:  

  void organizeByLayer(const reco::HGCalMultiCluster::ClusterCollection &);
  void reset(){
    for( std::vector< std::vector<KDNode> >::iterator it = points.begin(); it != points.end(); it++)
      {
        it->clear();
      }
    for(unsigned int i = 0; i < minpos.size(); i++)
      {
	minpos[i][0]=0.;minpos[i][1]=0.;
	maxpos[i][0]=0.;maxpos[i][1]=0.;
      }
  }
  void layerIntersection(std::array<double,3> &to, const std::array<double,3> &from) const;

  //max number of layers
  static const unsigned int maxlayer = 52;

  float radius;
  uint32_t minClusters;
  struct ClusterRef {
    int ind;
    float z;
    ClusterRef(int ind_i, float z_i): ind(ind_i),z(z_i){}
    ClusterRef(): ind(-1),z(0.){}
  };

  typedef KDTreeLinkerAlgo<ClusterRef,2> KDTree;
  typedef KDTreeNodeInfoT<ClusterRef,2> KDNode;
  std::vector< std::vector<KDNode> > points;
  std::vector<std::array<float,2> > minpos;
  std::vector<std::array<float,2> > maxpos;
  std::vector<size_t> es;
  std::vector<float>zees;
  std::unique_ptr<hgcal::ClusterTools> clusterTools;
  hgcal::RecHitTools rhtools_;

};

#endif
