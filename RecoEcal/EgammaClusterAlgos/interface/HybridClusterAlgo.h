#ifndef RecoEcal_EgammaClusterAlgos_HybridClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_HybridClusterAlgo_h

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoEcal/EgammaClusterAlgos/interface/LogPositionCalc.h"
#include <vector>

struct less_mag : public std::binary_function<EcalRecHit, EcalRecHit, bool> {
  bool operator()(EcalRecHit x, EcalRecHit y) { return x.energy() < y.energy() ; }
};

class HybridClusterAlgo
{
 private:
  typedef math::XYZPoint Point;
  double eb_st, ec_st;
  int phi_steps;
  double Ethres, Ewing;
  // Map of DetId, bool is if Det has been 
  // used already.
  std::map<EBDetId, std::pair<EcalRecHit, bool> >  rechits_m;

  // The vector of seeds:
  std::vector<EcalRecHit> seeds;

 public:
  
  HybridClusterAlgo(){}
  
  //eb_st --> ECAL barrel seed threshold
  //ec_st --> ECAL endcap seed threshold
  //phi_steps-->  How many domino steps to go in phi (each direction)
  //Ethres--> domino energy threshold
  //Ewing -->  Threshold to add additional cells to domino
  //Eseed -->  Threshold to be a peak among dominos

  HybridClusterAlgo(double eb_str, 
		    double ec_str, 
		    int step, 
		    double ethresh, 
		    double ewing) : eb_st(eb_str), ec_st(ec_str), 
    phi_steps(step), Ethres(ethresh), Ewing(ewing)
  {
    
  }
  
  std::vector<reco::BasicCluster> makeClusters(EcalRecHitCollection & rechits, const CaloSubdetectorGeometry & geometry);
  std::vector<reco::BasicCluster> mainSearch(const CaloSubdetectorGeometry & geometry);
  double makeDomino(EcalBarrelNavigator &navigator, std::vector <EcalRecHit> &cells);
  friend Point getECALposition(std::vector<reco::EcalRecHitData> recHits, const CaloSubdetectorGeometry & geometry);//Position determination

};

#endif
