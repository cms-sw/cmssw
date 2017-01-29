#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <list>

namespace {
  std::vector<size_t> sorted_indices(const reco::HGCalMultiCluster::ClusterCollection& v) {
    
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    
    // sort indices based on comparing values in v
    std::sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return (*v[i1]) > (*v[i2]);});
    
    return idx;
  } 

  float dist2(const edm::Ptr<reco::BasicCluster> &a, 
	      const edm::Ptr<reco::BasicCluster> &b) {
    return reco::deltaR2(*a,*b);
  }  
}

std::vector<reco::HGCalMultiCluster> HGCalDepthPreClusterer::makePreClusters(const reco::HGCalMultiCluster::ClusterCollection &thecls) const {

  std::vector<reco::HGCalMultiCluster> thePreClusters;
  std::vector<size_t> es = sorted_indices(thecls);
  std::vector<int> vused(es.size(),0);
  unsigned int used = 0;
  const float radius2 = radius*radius;

  for(unsigned int i = 0; i < es.size(); ++i) {
    if(vused[i]==0) {
      reco::HGCalMultiCluster temp;      
      temp.push_back(thecls[es[i]]);
      vused[i]=(thecls[es[i]]->z()>0)? 1 : -1;
      ++used;
      for(unsigned int j = i+1; j < es.size(); ++j) {
	if(vused[j]==0) {
	  if( dist2(thecls[es[i]],thecls[es[j]]) < radius2 && int(thecls[es[i]]->z()*vused[i])>0 ) {
	    temp.push_back(thecls[es[j]]);
	    vused[j]=vused[i];
	    ++used;
	  }	
	}
      }
      if( temp.size() > minClusters ) {
        thePreClusters.push_back(temp);
        auto& back = thePreClusters.back();
        back.setPosition(clusterTools->getMultiClusterPosition(back));
        back.setEnergy(clusterTools->getMultiClusterEnergy(back));
      }
    }
  }
  
  

  return thePreClusters;
}



