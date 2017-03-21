#ifndef RecoParticleFlow_PFClusterProducer_PFEcalEndcapRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFEcalEndcapRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"


#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class PFEcalEndcapRecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFEcalEndcapRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
      srFlagToken_ = iC.consumes<EESrFlagCollection>(iConfig.getParameter<edm::InputTag>("srFlags"));
      elecMap_ = 0;
    }

    void importRecHits(std::unique_ptr<reco::PFRecHitCollection>&out,std::unique_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {

      beginEvent(iEvent,iSetup);
 
      edm::Handle<EcalRecHitCollection> recHitHandle;

      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
      iEvent.getByToken(srFlagToken_,srFlagHandle_);

      // get the ecal geometry
      const CaloSubdetectorGeometry *gTmp = 
	geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

      const EcalEndcapGeometry *ecalGeo =dynamic_cast< const EcalEndcapGeometry* > (gTmp);

      iEvent.getByToken(recHitToken_,recHitHandle);
      for(const auto& erh : *recHitHandle ) {      
	const DetId& detid = erh.detid();
	auto energy = erh.energy();
	auto time = erh.time();

        bool hi = isHighInterest(detid);
        
	const CaloCellGeometry * thisCell= ecalGeo->getGeometry(detid);
  
	// find rechit geometry
	if(!thisCell) {
	  edm::LogError("PFEcalEndcapRecHitCreator")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  continue;
	}

	out->emplace_back(thisCell, detid.rawId(), PFLayer::ECAL_ENDCAP, energy); 

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
      
      edm::ESHandle< EcalElectronicsMapping > ecalmapping;
      es.get< EcalMappingRcd >().get(ecalmapping);
      elecMap_ = ecalmapping.product();

    }

 protected:


    bool isHighInterest(const EEDetId& detid) {
      bool result=false;
      EESrFlagCollection::const_iterator srf = srFlagHandle_->find(readOutUnitOf(detid));
      if(srf==srFlagHandle_->end()) return false;
      else result = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK) == EcalSrFlag::SRF_FULL);
      return result;
    }

    EcalScDetId readOutUnitOf(const EEDetId& detid) const{
      const EcalElectronicsId& EcalElecId = elecMap_->getElectronicsId(detid);
      int iDCC= EcalElecId.dccId();
      int iDccChan = EcalElecId.towerId();
      const bool ignoreSingle = true;
      const std::vector<EcalScDetId> id = elecMap_->getEcalScDetId(iDCC, iDccChan, ignoreSingle);
      return id.size()>0?id[0]:EcalScDetId();
    }


    edm::EDGetTokenT<EcalRecHitCollection> recHitToken_;
    edm::EDGetTokenT<EESrFlagCollection> srFlagToken_;

    const EcalTrigTowerConstituentsMap* eTTmap_;  

    // Ecal electronics/geometrical mapping
    const EcalElectronicsMapping* elecMap_;
    // selective readout flags collection
    edm::Handle<EESrFlagCollection> srFlagHandle_;

};

#endif
