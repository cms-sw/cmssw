#ifndef RecoEcal_EgammaClusterAlgos_Multi5x5BremRecoveryClusterAlgo_h_
#define RecoEcal_EgammaClusterAlgos_Multi5x5BremRecoveryClusterAlgo_h_

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoEcal/EgammaCoreTools/interface/BremRecoveryPhiRoadAlgo.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>


/*
  The Multi5x5BremRecoveryClusterAlgo class encapsulates the functionality needed
  to perform the SuperClustering.
  
  WARNING: This code assumes that the BasicClusters 
  from the event are sorted by energy
*/

class Multi5x5BremRecoveryClusterAlgo
{
 public:
  

  Multi5x5BremRecoveryClusterAlgo(const edm::ParameterSet &bremRecoveryPset,
			  double eb_sc_road_etasize = 0.06, // Search window in eta - Barrel
			  double eb_sc_road_phisize = 0.80, // Search window in phi - Barrel
			  double ec_sc_road_etasize = 0.14, // Search window in eta - Endcap
			  double ec_sc_road_phisize = 0.40, // Search window in eta - Endcap
			  bool dynamicPhiRoad = true,
			  double theSeedTransverseEnergyThreshold = 0.40
			  )
    {
      // e*_rdeta_ and e*_rdphi_ are half the total window 
      // because they correspond to one direction (positive or negative)
      eb_rdeta_ = eb_sc_road_etasize / 2;
      eb_rdphi_ = eb_sc_road_phisize / 2;
      ec_rdeta_ = ec_sc_road_etasize / 2;
      ec_rdphi_ = ec_sc_road_phisize / 2;

      seedTransverseEnergyThreshold = theSeedTransverseEnergyThreshold;
      dynamicPhiRoad_ = dynamicPhiRoad;
      if (dynamicPhiRoad_) phiRoadAlgo_ = new BremRecoveryPhiRoadAlgo(bremRecoveryPset);

    }

  // destructor
  ~Multi5x5BremRecoveryClusterAlgo() 
  {
     if (dynamicPhiRoad_) delete phiRoadAlgo_;
  } 

  
  // the method called from outside to do the SuperClustering - returns a vector of SCs:
  reco::SuperClusterCollection makeSuperClusters(reco::CaloClusterPtrVector & clusters);
  
 private:
  
  // make superclusters out of clusters produced by the Island algorithm:
  void makeIslandSuperClusters(reco::CaloClusterPtrVector &clusters_v, 
			       double etaRoad, double phiRoad);
  
 
  double eb_rdeta_;
  double eb_rdphi_;
  double ec_rdeta_;
  double ec_rdphi_;
  
  double seedTransverseEnergyThreshold;
  bool dynamicPhiRoad_;  
  BremRecoveryPhiRoadAlgo *phiRoadAlgo_;

  reco::SuperClusterCollection superclusters_v;
  
};

#endif
