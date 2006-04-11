#ifndef RecoECAL_ECALClusters_IslandClusterAlgo_h
#define RecoECAL_ECALClusters_IslandClusterAlgo_h

#include "RecoEcal/EgammaClusterAlgos/interface/ClusteringAlgorithm.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"


class IslandClusterAlgo : public ClusteringAlgorithm
{
 private:

  std::vector<reco::EcalRecHitData> hitData_v;

 public:

  IslandClusterAlgo() : ClusteringAlgorithm()
    {

    }
  
  IslandClusterAlgo(double ebst, double ecst) : ClusteringAlgorithm(ebst, ecst)
    {

    }
  
  void mainSearch(edm::ESHandle<CaloGeometry> geometry_h);

  void searchNorth(EcalBarrelNavigator &navigator);
  void searchSouth(EcalBarrelNavigator &navigator);
  void searchWest (EcalBarrelNavigator &navigator, EcalBarrelTopology &topology);
  void searchEast (EcalBarrelNavigator &navigator, EcalBarrelTopology &topology);

};

#endif
