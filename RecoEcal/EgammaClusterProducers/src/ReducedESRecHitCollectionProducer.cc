#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include "RecoEcal/EgammaClusterProducers/interface/ReducedESRecHitCollectionProducer.h"

ReducedESRecHitCollectionProducer::ReducedESRecHitCollectionProducer(const edm::ParameterSet& ps) {

 scEtThresh_          = ps.getParameter<double>("scEtThreshold");

 InputRecHitES_       = ps.getParameter<edm::InputTag>("EcalRecHitCollectionES");
 InputSpuerClusterEE_ = ps.getParameter<edm::InputTag>("EndcapSuperClusterCollection"); 

 OutputLabelES_       = ps.getParameter<std::string>("OutputLabel_ES");
 
 produces< EcalRecHitCollection > (OutputLabelES_);
 
}

ReducedESRecHitCollectionProducer::~ReducedESRecHitCollectionProducer() {
}

void ReducedESRecHitCollectionProducer::beginJob(){
}

void ReducedESRecHitCollectionProducer::endJob(){
}

void ReducedESRecHitCollectionProducer::produce(edm::Event & e, const edm::EventSetup& iSetup){

  ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;

  CaloSubdetectorTopology *topology_p = 0;
  if (geometry) topology_p = new EcalPreshowerTopology(geoHandle);

  edm::Handle<ESRecHitCollection> ESRecHits_;
  e.getByLabel(InputRecHitES_, ESRecHits_);
  
  std::auto_ptr<EcalRecHitCollection> ReducedESRecHitCollectionProducer(new EcalRecHitCollection);

  // make the map of rechits
  rechits_map_.clear();
  used_strips_.clear();
  EcalRecHitCollection::const_iterator it;
  if (ESRecHits_.isValid()) {
    for (it = ESRecHits_->begin(); it != ESRecHits_->end(); ++it) {
      if (it->recoFlag()==1 || it->recoFlag()==14 || (it->recoFlag()<=10 && it->recoFlag()>=5)) continue;
      rechits_map_.insert(std::make_pair(it->id(), *it));   
      //cout<<"RH : "<<ESDetId(it->id())<<" "<<it->energy()<<endl;
    }
  }
  
  edm::Handle<reco::SuperClusterCollection> pEndcapSuperClusters;
  if (e.getByLabel(InputSpuerClusterEE_, pEndcapSuperClusters)) {
    
    const reco::SuperClusterCollection* eeSuperClusters = pEndcapSuperClusters.product();
    
    for (reco::SuperClusterCollection::const_iterator isc = eeSuperClusters->begin(); isc != eeSuperClusters->end(); ++isc) {

      if (isc->energy() < scEtThresh_) continue;
      if (fabs(isc->eta()) < 1.65 || fabs(isc->eta()) > 2.6) continue;
      //cout<<"SC energy : "<<isc->energy()<<" "<<isc->eta()<<endl;

      //Int_t nBC = 0;
      reco::CaloCluster_iterator ibc = isc->clustersBegin();
      for ( ; ibc != isc->clustersEnd(); ++ibc ) {  

	//cout<<"BC : "<<nBC<<endl;

	double X = (*ibc)->x();
	double Y = (*ibc)->y();
	double Z = (*ibc)->z();        
	
	EcalRecHitCollection ESHits0 = getESHits(X, Y, Z, geometry_p, topology_p, 0);
	for (it = ESHits0.begin(); it != ESHits0.end(); ++it) {
	  ReducedESRecHitCollectionProducer->push_back(*it);
	  used_strips_.insert(std::make_pair(it->id(), 1));
	}
	EcalRecHitCollection ESHits1 = getESHits(X, Y, Z, geometry_p, topology_p, 1);
	for (it = ESHits1.begin(); it != ESHits1.end(); ++it) {
	  ReducedESRecHitCollectionProducer->push_back(*it);
	  used_strips_.insert(std::make_pair(it->id(), 1));
	}
	EcalRecHitCollection ESHits2 = getESHits(X, Y, Z, geometry_p, topology_p, -1);
	for (it = ESHits2.begin(); it != ESHits2.end(); ++it) {
	  ReducedESRecHitCollectionProducer->push_back(*it);
	  used_strips_.insert(std::make_pair(it->id(), 1));
	}
	
	//nBC++;
      }
      
    }
    
  }
  
  e.put(ReducedESRecHitCollectionProducer, OutputLabelES_);
  
}

EcalRecHitCollection ReducedESRecHitCollectionProducer::getESHits(double X, double Y, double Z, const CaloSubdetectorGeometry*& geometry_p, CaloSubdetectorTopology *topology_p, int row) {

  //cout<<row<<endl;

  EcalRecHitCollection esHits;
  int used = 0;

  const GlobalPoint point(X,Y,Z);

  DetId esId1 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 1);
  DetId esId2 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 2);
  ESDetId esDetId1 = (esId1 == DetId(0)) ? ESDetId(0) : ESDetId(esId1);
  ESDetId esDetId2 = (esId2 == DetId(0)) ? ESDetId(0) : ESDetId(esId2);  

  map<DetId, EcalRecHit>::iterator it;
  map<DetId, int>::iterator itu;
  ESDetId next;
  ESDetId strip1;
  ESDetId strip2;

  strip1 = esDetId1;
  strip2 = esDetId2;

  EcalPreshowerNavigator theESNav1(strip1, topology_p);
  theESNav1.setHome(strip1);
  
  EcalPreshowerNavigator theESNav2(strip2, topology_p);
  theESNav2.setHome(strip2);

  if (row == 1) {
    strip1 = theESNav1.north();
    strip2 = theESNav2.east();
  } else if (row == -1) {
    strip1 = theESNav1.south();
    strip2 = theESNav2.west();
  }

  // Plane 1 
  if (strip1 == ESDetId(0)) {
  } else {
    
    it = rechits_map_.find(strip1);
    itu = used_strips_.find(strip1);
    used = itu->second;
    //cout<<"center : "<<strip1<<" "<<it->second.energy()<<endl;      
    if (it != rechits_map_.end() && used == 0) {
      esHits.push_back(it->second);  
      //cout<<"Found !"<<endl;
    }

    // east road 
    for (int i=0; i<15; ++i) {
      next = theESNav1.east();
      if (next != ESDetId(0)) {
	it = rechits_map_.find(next);
	itu = used_strips_.find(next);
	used = itu->second;
	//cout<<"east "<<i<<" : "<<next<<" "<<it->second.energy()<<endl;
	if (it != rechits_map_.end() && used == 0) {
	  esHits.push_back(it->second);  
	  //cout<<"Found !"<<endl;
	}
      } else {
	break;
      }
    }

    // west road 
    theESNav1.setHome(strip1);
    theESNav1.home();
    for (int i=0; i<15; ++i) {
      next = theESNav1.west();
      if (next != ESDetId(0)) {
	it = rechits_map_.find(next);
	itu = used_strips_.find(next);
	used = itu->second;
	//cout<<"west "<<i<<" : "<<next<<" "<<it->second.energy()<<endl;
	if (it != rechits_map_.end() && used == 0) {
	  esHits.push_back(it->second);  
	  //cout<<"Found !"<<endl;
	}
      } else {
	break;
      }
    }

  }

  if (strip2 == ESDetId(0)) {
  } else {

    it = rechits_map_.find(strip2);
    itu = used_strips_.find(strip2);
    used = itu->second;
    //cout<<"center : "<<strip2<<" "<<it->second.energy()<<endl;      
    if (it != rechits_map_.end() && used == 0) {
      esHits.push_back(it->second);
      //cout<<"Found !"<<endl;
    }

    // north road 
    for (int i=0; i<15; ++i) {
      next = theESNav2.north();
      if (next != ESDetId(0)) {
	it = rechits_map_.find(next);
	itu = used_strips_.find(next);
	used = itu->second;
	//cout<<"north "<<i<<" : "<<next<<" "<<it->second.energy()<<endl;  
	if (it != rechits_map_.end() && used == 0) {
	  esHits.push_back(it->second);
	  //cout<<"Found !"<<endl;
	}
      } else {
	break;
      } 
    }

    // south road 
    theESNav2.setHome(strip2);
    theESNav2.home();
    for (int i=0; i<15; ++i) {
      next = theESNav2.south();
      if (next != ESDetId(0)) {
	it = rechits_map_.find(next);
	itu = used_strips_.find(next);
	used = itu->second;
	//cout<<"south "<<i<<" : "<<next<<" "<<it->second.energy()<<endl;
	if (it != rechits_map_.end() && used == 0) {
	  esHits.push_back(it->second);
	  //cout<<"Found !"<<endl;
	}
      } else {
	break;
      }
    }
  }
  
  return esHits;
}
