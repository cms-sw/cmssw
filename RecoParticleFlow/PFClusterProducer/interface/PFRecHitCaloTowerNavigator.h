#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitCaloTowerNavigator_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitCaloTowerNavigator_h


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"

//ARGH Need to inherit the mother function to initialize the topology 
//since the CT topology doesnt use geometry as input
class PFRecHitCaloTowerNavigator : public PFRecHitNavigatorBase {
 public:
  PFRecHitCaloTowerNavigator() {

  }



  PFRecHitCaloTowerNavigator(const edm::ParameterSet& iConfig)
    {

    }

  void beginEvent(const edm::EventSetup& iSetup) {
      topology_ = new CaloTowerTopology();
  }

  void associateNeighbours(reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits) {
      CaloTowerDetId detid( hit.detId() );
      
      CaloNavigator<CaloTowerDetId> navigator(detid, topology_);
      
      DetId N(0);
      DetId E(0);
      DetId S(0);
      DetId W(0);
      DetId NW(0);
      DetId NE(0);
      DetId SW(0);
      DetId SE(0);


      N=navigator.north();  
      associateNeighbour(N,hit,hits,true);


      if (N !=DetId(0)) {
	NE=navigator.east();
      }
      else 
	{
	  navigator.home();
	  E=navigator.east();
	  NE=navigator.north();
	}
      associateNeighbour(NE,hit,hits,1,1,0);
      navigator.home();

      S = navigator.south();
      associateNeighbour(S,hit,hits,true);
      
      if (S !=DetId(0)) {
	SW = navigator.west();
      } else {
	navigator.home();
	W=navigator.west();
	SW=navigator.south();
      }
      associateNeighbour(SW,hit,hits,false);
      navigator.home();

      E = navigator.east();
      associateNeighbour(E,hit,hits,true);
      
      if (E !=DetId(0)) {
	SE = navigator.south();
      } else {
	navigator.home();
	S=navigator.south();
	SE=navigator.east();
      }
      associateNeighbour(SE,hit,hits,false);
      navigator.home();


      W = navigator.west();
      associateNeighbour(W,hit,hits,true);

      if (W !=DetId(0)) {
	NW = navigator.north();
      } else {
	navigator.home();
	N=navigator.north();
	NW=navigator.west();
      }
      associateNeighbour(NW,hit,hits,false);
  }



 protected:
  CaloTowerTopology *topology_;


};


#endif
