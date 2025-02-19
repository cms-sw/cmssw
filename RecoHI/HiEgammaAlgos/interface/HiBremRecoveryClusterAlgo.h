#ifndef RecoHi_HiEgammaAlgos_HiBremRecoveryClusterAlgo_h_
#define RecoHi_HiEgammaAlgos_HiBremRecoveryClusterAlgo_h_

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include <vector>


/*
  The HiBremRecoveryClusterAlgo class encapsulates the functionality needed
  to perform the SuperClustering.
  
  WARNING: This code assumes that the BasicClusters 
  from the event are sorted by energy
*/

class HiBremRecoveryClusterAlgo
{
 public:
  
  enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 }; 

  HiBremRecoveryClusterAlgo(double eb_sc_road_etasize = 0.06, // Search window in eta - Barrel
			  double eb_sc_road_phisize = 0.80, // Search window in phi - Barrel
			  double ec_sc_road_etasize = 0.14, // Search window in eta - Endcap
			  double ec_sc_road_phisize = 0.40, // Search window in eta - Endcap
			  double theSeedTransverseEnergyThreshold = 0.40,
			  double theBarrelBremEnergyThreshold = 2.3,
			  double theEndcapBremEnergyThreshold = 5.7,
			  VerbosityLevel the_verbosity = pERROR
			  )
  {
      // e*_rdeta_ and e*_rdphi_ are half the total window 
      // because they correspond to one direction (positive or negative)
      eb_rdeta_ = eb_sc_road_etasize / 2;
      eb_rdphi_ = eb_sc_road_phisize / 2;
      ec_rdeta_ = ec_sc_road_etasize / 2;
      ec_rdphi_ = ec_sc_road_phisize / 2;

      seedTransverseEnergyThreshold = theSeedTransverseEnergyThreshold;
      BarrelBremEnergyThreshold = theBarrelBremEnergyThreshold;
      EndcapBremEnergyThreshold = theEndcapBremEnergyThreshold;
      verbosity = the_verbosity;
  }

  void setVerbosity(VerbosityLevel the_verbosity)
  {
      verbosity = the_verbosity;
  }
 
  // the method called from outside to do the SuperClustering - returns a vector of SCs:
  reco::SuperClusterCollection makeSuperClusters(reco::CaloClusterPtrVector & clusters);
  
 private:
  
  // make superclusters out of clusters produced by the Island algorithm:
  void makeIslandSuperClusters(reco::CaloClusterPtrVector &clusters_v, 
			       double etaRoad, double phiRoad);
  
  // return true if the cluster is within the search phi-eta window of the seed
  bool match(reco::CaloClusterPtr seed_p, 
	     reco::CaloClusterPtr cluster_p,
	     double etaRoad, double phiRoad);
  
  VerbosityLevel verbosity;

  double eb_rdeta_;
  double eb_rdphi_;
  double ec_rdeta_;
  double ec_rdphi_;
  
  double seedTransverseEnergyThreshold;

  // Barrel Basic Cluster threshold in the brem recovery process
  double BarrelBremEnergyThreshold;
  double EndcapBremEnergyThreshold;
  
  reco::SuperClusterCollection superclusters_v;
  
};

#endif
