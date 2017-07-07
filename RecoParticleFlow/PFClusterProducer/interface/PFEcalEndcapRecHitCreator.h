#ifndef RecoParticleFlow_PFClusterProducer_PFEcalEndcapRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFEcalEndcapRecHitCreator_h

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

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
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
      auto srF = iConfig.getParameter<edm::InputTag>("srFlags");
      if (not srF.label().empty())
	srFlagToken_ = iC.consumes<EESrFlagCollection>(srF);
      elecMap_ = nullptr;
    }

  void importRecHits(std::unique_ptr<reco::PFRecHitCollection>&out,std::unique_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) override {

    beginEvent(iEvent,iSetup);
 
    edm::Handle<EcalRecHitCollection> recHitHandle;

    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
    bool useSrF = false;
    if (not srFlagToken_.isUninitialized()){
      iEvent.getByToken(srFlagToken_,srFlagHandle_);
      useSrF = true;
    }

    // get the ecal geometry
    const CaloSubdetectorGeometry *gTmp = 
      geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

    const EcalEndcapGeometry *ecalGeo =dynamic_cast< const EcalEndcapGeometry* > (gTmp);

    iEvent.getByToken(recHitToken_,recHitHandle);
    for(const auto& erh : *recHitHandle ) {      
      const DetId& detid = erh.detid();
      auto energy = erh.energy();
      auto time = erh.time();

      bool hi = (useSrF ? isHighInterest(detid) : true);
        
      const CaloCellGeometry * thisCell= ecalGeo->getGeometry(detid);
  
      // find rechit geometry
      if(!thisCell) {
        throw cms::Exception("PFEcalEndcapRecHitCreator") 
          << "detid "<< detid.rawId() << "not found in geometry";
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

  void init(const edm::EventSetup &es) override {
      
    edm::ESHandle< EcalElectronicsMapping > ecalmapping;
    es.get< EcalMappingRcd >().get(ecalmapping);
    elecMap_ = ecalmapping.product();

  }

 protected:


  bool isHighInterest(const EEDetId& detid) {
    bool result=false;
    auto srf = srFlagHandle_->find(readOutUnitOf(detid));
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
