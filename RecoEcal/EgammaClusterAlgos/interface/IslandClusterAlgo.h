#ifndef RecoECAL_ECALClusters_IslandClusterAlgo_h
#define RecoECAL_ECALClusters_IslandClusterAlgo_h

#include "RecoEcal/EgammaClusterAlgos/interface/ClusteringAlgorithm.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"


class IslandClusterAlgo : public ClusteringAlgorithm
{
 private:

  std::vector<reco::EcalRecHitData> hitData_v;

 public:

  struct ClusterVars{
    double energy;
    double chi2;
    std::vector<DetId> usedHits;
  };

  IslandClusterAlgo() : ClusteringAlgorithm()
    {

    }
  
  IslandClusterAlgo(double ebst, double ecst) : ClusteringAlgorithm(ebst, ecst)
    {

    }
  
  void mainSearch(const CaloSubdetectorGeometry &geometry);

  void searchNorth(EcalBarrelNavigator &navigator);
  void searchSouth(EcalBarrelNavigator &navigator);
  void searchWest (EcalBarrelNavigator &navigator, EcalBarrelHardcodedTopology &topology);
  void searchEast (EcalBarrelNavigator &navigator, EcalBarrelHardcodedTopology &topology);

  ClusterVars computeClusterVars( const std::vector<reco::EcalRecHitData>& hits ) const;


};

#endif
