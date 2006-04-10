#ifndef RecoEcal_EcalClusterAlgos_BremRecoveryClusterAlgo_h_
#define RecoEcal_EcalClusterAlgos_BremRecoveryClusterAlgo_h_

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoEcal/EgammaClusterAlgos/interface/SuperCluster.h"

#include <vector>

/*
  The BremRecoveryClusterAlgo class encapsulates the functionality needed
  to perform the SuperClustering.

  WARNING: This code assumes that the BasicClusters 
  from the event are sorted by energy
 */

class BremRecoveryClusterAlgo
{
 public:

  BremRecoveryClusterAlgo(double eb_sc_road_etasize = 0.06, // Search window in eta - Barrel
	       double eb_sc_road_phisize = 0.80, // Search window in phi - Barrel
	       double ec_sc_road_etasize = 0.14, // Search window in eta - Endcap
	       double ec_sc_road_phisize = 0.40, // Search window in eta - Endcap
	       double theSeedEnergyThreshold = 0.40)
    {
      eb_rdeta_ = eb_sc_road_etasize;
      eb_rdphi_ = eb_sc_road_phisize;
      ec_rdeta_ = ec_sc_road_etasize;
      ec_rdphi_ = ec_sc_road_phisize;
      seedEnergyThreshold = theSeedEnergyThreshold;
    }
  
  // the method called from outside to do the SuperClustering - returns a vector of SCs:
  std::vector<SuperCluster> makeSuperClusters(reco::BasicClusterCollection & clusters);
 
 private:
  
  // make superclusters out of clusters produced by the Island algorithm:
  void makeIslandSuperClusters(std::vector<reco::BasicCluster *> &clusters_v, 
			       double etaRoad, double phiRoad);

  // make superclusters out of clusters produced by the Hybrid algorithm:
  void makeHybridSuperClusters(std::vector<reco::BasicCluster *> &clusters_v);

  // return true if the cluster is within the search phi-eta window of the seed
  bool match(reco::BasicCluster *seed_p, 
	     reco::BasicCluster *cluster_p,
	     double etaRoad, double phiRoad);
 

  double eb_rdeta_;
  double eb_rdphi_;
  double ec_rdeta_;
  double ec_rdphi_;

  double seedEnergyThreshold;

  std::vector<SuperCluster> superclusters_v;
  
};

#endif
