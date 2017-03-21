#ifndef RecoParticleFlow_PFClusterProducer_PFEcalBarrelRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFEcalBarrelRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class PFEcalBarrelRecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFEcalBarrelRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
      srFlagToken_ = iC.consumes<EBSrFlagCollection>(iConfig.getParameter<edm::InputTag>("srFlags"));
      triggerTowerMap_ = 0;
    }
    
    void importRecHits(std::unique_ptr<reco::PFRecHitCollection>&out,std::unique_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {

      beginEvent(iEvent,iSetup);
      
      edm::Handle<EcalRecHitCollection> recHitHandle;

      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
      iEvent.getByToken(srFlagToken_,srFlagHandle_);

      // get the ecal geometry
      const CaloSubdetectorGeometry *gTmp = 
	geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);

      const EcalBarrelGeometry *ecalGeo =dynamic_cast< const EcalBarrelGeometry* > (gTmp);

      iEvent.getByToken(recHitToken_,recHitHandle);
      for(const auto& erh : *recHitHandle ) {      
	const DetId& detid = erh.detid();
	auto energy = erh.energy();
	auto time = erh.time();
        bool hi = isHighInterest(detid);

	const CaloCellGeometry * thisCell= ecalGeo->getGeometry(detid);
  
	// find rechit geometry
	if(!thisCell) {
	  edm::LogError("PFEcalBarrelRecHitCreator")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  continue;
	}

	out->emplace_back(thisCell, detid.rawId(), PFLayer::ECAL_BARREL, energy); 

        auto & rh = out->back();
	
	bool rcleaned = false;
	bool keep=true;

	//Apply Q tests
	for( const auto& qtest : qualityTests_ ) {
	  if (!qtest->test(rh,erh,rcleaned,hi)) {
	    keep = false;	    
	  }
	}
	  
	if(keep) {
	  rh.setTime(time);
	  rh.setDepth(1);
	} 
        else {
	  if (rcleaned) 
	    cleaned->push_back(std::move(out->back()));
          out->pop_back();
        }
      }
    }

    void init(const edm::EventSetup &es) {

      edm::ESHandle<EcalTrigTowerConstituentsMap> hTriggerTowerMap;
      es.get<IdealGeometryRecord>().get(hTriggerTowerMap);
      triggerTowerMap_ = hTriggerTowerMap.product();

    }
      

 protected:

    bool isHighInterest(const EBDetId& detid) {
      bool result=false;
      EBSrFlagCollection::const_iterator srf = srFlagHandle_->find(readOutUnitOf(detid));
      if(srf==srFlagHandle_->end()) return false;
      else result = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK) == EcalSrFlag::SRF_FULL);
      return result;
    }

    EcalTrigTowerDetId readOutUnitOf(const EBDetId& detid) const{
      return triggerTowerMap_->towerOf(detid);
    }

    edm::EDGetTokenT<EcalRecHitCollection> recHitToken_;
    edm::EDGetTokenT<EBSrFlagCollection> srFlagToken_;

    // ECAL trigger tower mapping
    const EcalTrigTowerConstituentsMap * triggerTowerMap_;
    // selective readout flags collection
    edm::Handle<EBSrFlagCollection> srFlagHandle_;

};

#endif
