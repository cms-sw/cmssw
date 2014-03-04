#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitCaloNavigatorWithTime_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitCaloNavigatorWithTime_h


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


template <typename D,typename T>
class PFRecHitCaloNavigatorWithTime : public PFRecHitNavigatorBase {
 public:
  PFRecHitCaloNavigatorWithTime(const edm::ParameterSet& iConfig) {
    binning_ = iConfig.getParameter<std::vector<double> >("timeDigitization");
  }

  void associateNeighbours(reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refProd) {
      DetId detid( hit.detId() );
      
      CaloNavigator<D> navigator(detid, topology_);
      
      DetId N(0);
      DetId E(0);
      DetId S(0);
      DetId W(0);
      DetId NW(0);
      DetId NE(0);
      DetId SW(0);
      DetId SE(0);


      N=navigator.north();  
      associateNeighbour(N,hit,hits,refProd,0,1);


      if (N !=DetId(0)) {
	NE=navigator.east();
      }
      else 
	{
	  navigator.home();
	  E=navigator.east();
	  NE=navigator.north();
	}
      associateNeighbour(NE,hit,hits,refProd,1,1);
      navigator.home();

      S = navigator.south();
      associateNeighbour(S,hit,hits,refProd,0,-1);
      
      if (S !=DetId(0)) {
	SW = navigator.west();
      } else {
	navigator.home();
	W=navigator.west();
	SW=navigator.south();
      }
      associateNeighbour(SW,hit,hits,refProd,-1,-1);
      navigator.home();

      E = navigator.east();
      associateNeighbour(E,hit,hits,refProd,1,0);
      
      if (E !=DetId(0)) {
	SE = navigator.south();
      } else {
	navigator.home();
	S=navigator.south();
	SE=navigator.east();
      }
      associateNeighbour(SE,hit,hits,refProd,1,-1);
      navigator.home();


      W = navigator.west();
      associateNeighbour(W,hit,hits,refProd,-1,0);

      if (W !=DetId(0)) {
	NW = navigator.north();
      } else {
	navigator.home();
	N=navigator.north();
	NW=navigator.west();
      }
      associateNeighbour(NW,hit,hits,refProd,-1,1);
  }



 protected:

  unsigned int get_bin(float time) {
    for (unsigned int i=0;i<binning_.size();++i) {
      if (i==0) {
	if (time<=binning_[0])
	  return 0;
      }
      else if (i==binning_[binning_.size()-1]) {
	if (time>binning_[binning_.size()-1])
	  return binning_.size()-1;
      }
      else {
	if (time>binning_[i-1] && time<=binning_[i] )
	  return i;
      }
    }
    return -9999;
  }



  void associateNeighbour(const DetId& id, reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refProd,short eta, short phi) {
    int depth=0;
    const reco::PFRecHit temp(id,PFLayer::NONE,0.0,math::XYZPoint(0,0,0),math::XYZVector(0,0,0),std::vector<math::XYZPoint>());
    auto found_hit = std::lower_bound(hits->begin(),hits->end(),
				      temp,
				      [](const reco::PFRecHit& a, 
					 const reco::PFRecHit& b){
					return a.detId() < b.detId();
				      });
    if( found_hit != hits->end() && found_hit->detId() == id.rawId() ) {
      depth = get_bin(hit.time())-get_bin(found_hit->time());
      if( std::abs(depth) <= 1 ) {
	hit.addNeighbour(eta,phi,depth,reco::PFRecHitRef(refProd,std::distance(hits->begin(),found_hit)));
      }
    }
  }



  const T *topology_;
  std::vector<double> binning_;



};

#endif


