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
#include "DataFormats/DetId/interface/DetIdCollection.h"

ReducedESRecHitCollectionProducer::ReducedESRecHitCollectionProducer(const edm::ParameterSet& ps):
  geometry_p(0),
  topology_p(0)
{

 scEtThresh_          = ps.getParameter<double>("scEtThreshold");

 InputRecHitES_       = ps.getParameter<edm::InputTag>("EcalRecHitCollectionES");
 InputSpuerClusterEE_ = ps.getParameter<edm::InputTag>("EndcapSuperClusterCollection"); 

 OutputLabelES_       = ps.getParameter<std::string>("OutputLabel_ES");
 
 interestingDetIdCollections_         = ps.getParameter<std::vector< edm::InputTag> >("interestingDetIds");
 
 produces< EcalRecHitCollection > (OutputLabelES_);
 
}

ReducedESRecHitCollectionProducer::~ReducedESRecHitCollectionProducer() {
  if (topology_p) delete topology_p;
}

void ReducedESRecHitCollectionProducer::beginRun (edm::Run &, const edm::EventSetup&iSetup){
  ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  geometry_p = dynamic_cast<const EcalPreshowerGeometry *>(geometry);
  if (!geometry_p){
    edm::LogError("WrongGeometry")<<
      "could not cast the subdet geometry to preshower geometry";
  }
  
  if (geometry) topology_p = new EcalPreshowerTopology(geoHandle);
  
}

void ReducedESRecHitCollectionProducer::produce(edm::Event & e, const edm::EventSetup& iSetup){


  edm::Handle<ESRecHitCollection> ESRecHits_;
  e.getByLabel(InputRecHitES_, ESRecHits_);
  
  std::auto_ptr<EcalRecHitCollection> output(new EcalRecHitCollection);

  edm::Handle<reco::SuperClusterCollection> pEndcapSuperClusters;
  e.getByLabel(InputSpuerClusterEE_, pEndcapSuperClusters);
  {
    const reco::SuperClusterCollection* eeSuperClusters = pEndcapSuperClusters.product();
    
    for (reco::SuperClusterCollection::const_iterator isc = eeSuperClusters->begin(); isc != eeSuperClusters->end(); ++isc) {

      if (isc->energy() < scEtThresh_) continue;
      if (fabs(isc->eta()) < 1.65 || fabs(isc->eta()) > 2.6) continue;
      //cout<<"SC energy : "<<isc->energy()<<" "<<isc->eta()<<endl;

      //Int_t nBC = 0;
      reco::CaloCluster_iterator ibc = isc->clustersBegin();
      for ( ; ibc != isc->clustersEnd(); ++ibc ) {  

	//cout<<"BC : "<<nBC<<endl;

	const GlobalPoint point((*ibc)->x(),(*ibc)->y(),(*ibc)->z());
	
	collectIds(point, 0);
	collectIds(point, 1);
	collectIds(point, -1);

	//nBC++;
      }
      
    }
    
  }


  edm::Handle<DetIdCollection > detId;
  for( unsigned int t = 0; t < interestingDetIdCollections_.size(); ++t )
    {
      e.getByLabel(interestingDetIdCollections_[t],detId);
      if (!detId.isValid()){
	edm::LogError("MissingInput")<<"the collection of interesting detIds:"<<interestingDetIdCollections_[t]<<" is not found.";
        continue;
      }
      collectedIds_.insert(detId->begin(),detId->end());
    }


  output->reserve( collectedIds_.size());
  EcalRecHitCollection::const_iterator it;
  for (it = ESRecHits_->begin(); it != ESRecHits_->end(); ++it) {
    if (it->recoFlag()==1 || it->recoFlag()==14 || (it->recoFlag()<=10 && it->recoFlag()>=5)) continue;
    if (collectedIds_.find(it->id())!=collectedIds_.end()){
      output->push_back(*it);
    }
  }
  collectedIds_.clear();

  e.put(output, OutputLabelES_);

}

void ReducedESRecHitCollectionProducer::collectIds(const GlobalPoint & point,
						   const int & row){
  //cout<<row<<endl;

  DetId esId1 = geometry_p->getClosestCellInPlane(point, 1);
  DetId esId2 = geometry_p->getClosestCellInPlane(point, 2);
  ESDetId esDetId1 = (esId1 == DetId(0)) ? ESDetId(0) : ESDetId(esId1);
  ESDetId esDetId2 = (esId2 == DetId(0)) ? ESDetId(0) : ESDetId(esId2);  

  map<DetId,const EcalRecHit*>::iterator it;
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
    collectedIds_.insert(strip1);
    // east road 
    for (int i=0; i<15; ++i) {
      next = theESNav1.east();
      if (next != ESDetId(0)) {
	collectedIds_.insert(next);
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
	collectedIds_.insert(next);
      } else {
	break;
      }
    }

  }

  if (strip2 == ESDetId(0)) {
  } else {
    collectedIds_.insert(strip2);
    // north road 
    for (int i=0; i<15; ++i) {
      next = theESNav2.north();
      if (next != ESDetId(0)) {
	collectedIds_.insert(next);
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
	collectedIds_.insert(next);
      } else {
	break;
      }
    }
  }
}


