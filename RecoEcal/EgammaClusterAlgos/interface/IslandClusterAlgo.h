#ifndef RecoEcal_EcalClusterAlgos_IslandClusterAlgo_h
#define RecoEcal_EcalClusterAlgos_IslandClusterAlgo_h

#include "RecoEcal/EgammaClusterAlgos/interface/ClusteringAlgorithm.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoCaloTools/Navigation/interface/EBDetIdNavigator.h"


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
  
  void mainSearch(const CaloSubdetectorGeometry & geometry);

  void searchNorth(EBDetIdNavigator &navigator);
  void searchSouth(EBDetIdNavigator &navigator);
  void searchWest (EBDetIdNavigator &navigator);
  void searchEast (EBDetIdNavigator &navigator);

};

#endif
