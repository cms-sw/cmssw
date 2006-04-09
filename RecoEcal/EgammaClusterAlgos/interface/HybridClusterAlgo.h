#ifndef RecoECAL_ECALClusters_HybridClusterAlgo_h
#define RecoECAL_ECALClusters_HybridClusterAlgo_h

#include "RecoECAL/ECALClusters/interface/ClusteringAlgorithm.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoCaloTools/Navigation/interface/EBDetIdNavigator.h"
#include <vector>

class HybridClusterAlgo : public ClusteringAlgorithm
{
 private:

  reco::BasicCluster current_cluster;
  int phi_steps;
  double Ethres, Eseed, Ewing;

 public:

  HybridClusterAlgo() : ClusteringAlgorithm()
    {

    }
  
  //eb_st --> ECAL barrel seed threshold
  //ec_st --> ECAL endcap seed threshold
  //phi_steps-->  How many domino steps to go in phi (each direction)
  //Ethres--> domino energy threshold
  //Ewing -->  Threshold to add additional cells to domino
  //Eseed -->  Threshold to be a peak among dominos

  HybridClusterAlgo(double eb_st, 
	 double ec_st, 
	 int step, 
	 double ethresh, 
	 double eseed,
	 double ewing) : ClusteringAlgorithm(eb_st, ec_st), 
    phi_steps(step), Ethres(ethresh), Eseed(eseed), Ewing(ewing)
    {

    }
  
  void mainSearch(const CaloSubdetectorGeometry & geometry);
  double makeDomino(EBDetIdNavigator &navigator, std::vector <PositionAwareHit> &cells);


};

#endif
