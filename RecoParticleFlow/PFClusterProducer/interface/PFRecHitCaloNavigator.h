#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitCaloNavigator_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitCaloNavigator_h


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"


template <typename DET,typename TOPO,bool ownsTopo=true>
class PFRecHitCaloNavigator : public PFRecHitNavigatorBase {
 public:

 virtual ~PFRecHitCaloNavigator() { if(!ownsTopo) { topology_.release(); } }

  void associateNeighbours(reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refProd) {
      DetId detid( hit.detId() );
      
      CaloNavigator<DET> navigator(detid, topology_.get());
      
      DetId N(0);
      DetId E(0);
      DetId S(0);
      DetId W(0);
      DetId NW(0);
      DetId NE(0);
      DetId SW(0);
      DetId SE(0);


      N=navigator.north();  
      associateNeighbour(N,hit,hits,refProd,0,1,0);


      if (N !=DetId(0)) {
	NE=navigator.east();
      }
      else 
	{
	  navigator.home();
	  E=navigator.east();
	  NE=navigator.north();
	}
      associateNeighbour(NE,hit,hits,refProd,1,1,0);
      navigator.home();

      S = navigator.south();
      associateNeighbour(S,hit,hits,refProd,0,-1,0);
      
      if (S !=DetId(0)) {
	SW = navigator.west();
      } else {
	navigator.home();
	W=navigator.west();
	SW=navigator.south();
      }
      associateNeighbour(SW,hit,hits,refProd,-1,-1,0);
      navigator.home();

      E = navigator.east();
      associateNeighbour(E,hit,hits,refProd,1,0,0);
      
      if (E !=DetId(0)) {
	SE = navigator.south();
      } else {
	navigator.home();
	S=navigator.south();
	SE=navigator.east();
      }
      associateNeighbour(SE,hit,hits,refProd,1,-1,0);
      navigator.home();


      W = navigator.west();
      associateNeighbour(W,hit,hits,refProd,-1,0,0);

      if (W !=DetId(0)) {
	NW = navigator.north();
      } else {
	navigator.home();
	N=navigator.north();
	NW=navigator.west();
      }
      associateNeighbour(NW,hit,hits,refProd,-1,1,0);
  }



 protected:
  std::unique_ptr<const TOPO> topology_;


};

#endif


