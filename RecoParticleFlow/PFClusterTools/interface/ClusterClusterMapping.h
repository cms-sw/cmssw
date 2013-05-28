#ifndef CLUSTERCLUSTERMAPPING_H
#define CLUSTERCLUSTERMAPPING_H

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

class ClusterClusterMapping{
 public:
  ClusterClusterMapping(){;}
  ~ClusterClusterMapping(){;}
    
  // check the overlap of two CaloClusters
  static bool overlap(const reco::CaloCluster & sc1, const reco::CaloCluster & sc,float minfrac=0.01,bool debug=false) ;
  
  static int checkOverlap(const reco::PFCluster & pfc, const std::vector<const reco::SuperCluster *>& sc,float minfrac=0.01,bool debug=false) ;

  static int checkOverlap(const reco::PFCluster & pfc, const std::vector<reco::SuperClusterRef >& sc,float minfrac=0.01,bool debug=false) ;
};


#endif
