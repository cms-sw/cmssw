#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCalDepthPreClusterer.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCalImagingAlgo.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCalMultiCluster.h"

#include <list>

std::vector<HGCalMultiCluster> HGCalDepthPreClusterer::makePreClusters(const ClusterCollection &thecls){

  thePreClusters.clear();
  std::vector<size_t> es = sorted_indices<reco::BasicCluster>(thecls);
  std::vector<int> vused(es.size(),0);
  unsigned int used = 0;
  for(unsigned int i = 0; i < es.size(); i++){
    if(vused[i]==0){
      thePreClusters.push_back(HGCalMultiCluster(thecls[es[i]]));
      vused[i]=(thecls[es[i]].z()>0)? 1 : -1;
      used++;
      for(unsigned int j = i+1; j < es.size(); j++){
	if(vused[j]==0){
	  if(dist(thecls[es[i]],thecls[es[j]])<radius && int(thecls[es[i]].z()*vused[i])>0){
	    thePreClusters.back().push_back(thecls[es[j]]);
	    vused[j]=vused[i];
	    used++;
	  }	
	}
      }
    }
  }
  return thePreClusters;
}


float HGCalDepthPreClusterer::dist(const reco::BasicCluster &a, const reco::BasicCluster &b){
  return sqrt((a.eta()-b.eta())*(a.eta()-b.eta())+(a.phi()-b.phi())*(a.phi()-b.phi()));
}
