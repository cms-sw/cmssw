#ifndef RecoEcal_EgammaClusterAlgos_HybridClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_HybridClusterAlgo_h

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoEcal/EgammaClusterAlgos/interface/LogPositionCalc.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
  double Eseed;
  // Map of DetId, bool is if Det has been 
  // used already.
  std::map<EBDetId, std::pair<EcalRecHit, bool> >  rechits_m;

  // The vector of seeds:
  std::vector<EcalRecHit> seeds;
  std::map<int, std::vector<reco::BasicCluster> > _clustered;
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
		    double ewing,
		    double eseed) : eb_st(eb_str), ec_st(ec_str), 
    phi_steps(step), Ethres(ethresh), Ewing(ewing), Eseed(eseed)
  {
    
  }
  
  //  void makeClusters(EcalRecHitCollection & rechits, const CaloSubdetectorGeometry & geometry, reco::BasicClusterCollection &basicClusters);
  void makeClusters(EcalRecHitCollection & rechits, edm::ESHandle<CaloGeometry> , reco::BasicClusterCollection &basicClusters);

  reco::SuperClusterCollection makeSuperClusters(reco::BasicClusterRefVector);

  void mainSearch( const CaloSubdetectorGeometry geometry);
  double makeDomino(EcalBarrelNavigator &navigator, std::vector <EcalRecHit> &cells);

  friend Point getECALposition(std::vector<reco::EcalRecHitData> recHits,const CaloSubdetectorGeometry );//Position determination

};

#endif
