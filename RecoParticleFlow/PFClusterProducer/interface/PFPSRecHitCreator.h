#ifndef RecoParticleFlow_PFClusterProducer_PFPSRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFPSRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"


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
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class PFPSRecHitCreator final :  public  PFRecHitCreatorBase {

 public:  
  PFPSRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
    }

    void importRecHits(std::unique_ptr<reco::PFRecHitCollection>&out,std::unique_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) override {

      beginEvent(iEvent,iSetup);

      edm::Handle<EcalRecHitCollection> recHitHandle;
      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
      // get the ecal geometry
      const CaloSubdetectorGeometry *psGeometry = 
	geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

      iEvent.getByToken(recHitToken_,recHitHandle);
      for( const auto& erh : *recHitHandle ) {      
	ESDetId detid(erh.detid());
	auto energy = erh.energy();

	PFLayer::Layer layer = PFLayer::NONE;
	
	switch( detid.plane() ) {
	case 1:
	  layer = PFLayer::PS1;
	  break;
	case 2:
	  layer = PFLayer::PS2;
	  break;
	default:
	  throw cms::Exception("PFRecHitBadInput")
	    <<"incorrect preshower plane !! plane number "
	    <<detid.plane()<<std::endl;
	}
 

	
	const CaloCellGeometry * thisCell= psGeometry->getGeometry(detid);
  
	// find rechit geometry
	if(!thisCell) {
	  edm::LogError("PFPSRecHitCreator")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  continue;
	}

        out->emplace_back(thisCell, detid.rawId(),layer,energy);
        auto & rh = out->back();
	rh.setDepth(detid.plane());
	rh.setTime(erh.time());
	
	bool rcleaned = false;
	bool keep=true;
        bool hi = true; // all ES rhs are produced, independently on the ECAL SRP decision

	//Apply Q tests
	for( const auto& qtest : qualityTests_ ) {
	  if (!qtest->test(rh,erh,rcleaned,hi)) {
	    keep = false;	    
	  }
	}
	
        if (rcleaned) 
	  cleaned->push_back(std::move(out->back()));
        if(!keep) 
          out->pop_back();
      }
    }



 protected:
  edm::EDGetTokenT<EcalRecHitCollection> recHitToken_;


};


#endif
