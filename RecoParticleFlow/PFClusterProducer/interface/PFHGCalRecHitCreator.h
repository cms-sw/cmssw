#ifndef RecoParticleFlow_PFClusterProducer_PFHGCalRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFHGCalRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

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

template <typename DET,PFLayer::Layer Layer,unsigned subdet>
  class PFHGCalRecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFHGCalRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
      geometryInstance_ = iConfig.getParameter<std::string>("geometryInstance");
    }

    void importRecHits(std::unique_ptr<reco::PFRecHitCollection>&out,std::unique_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {

      for (unsigned int i=0;i<qualityTests_.size();++i) {
	qualityTests_.at(i)->beginEvent(iEvent,iSetup);
      }

      edm::Handle<HGCRecHitCollection> recHitHandle;
      iEvent.getByToken(recHitToken_,recHitHandle);
      const HGCRecHitCollection& rechits = *recHitHandle;

      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
      const CaloGeometry* geom = geoHandle.product();

      unsigned skipped_rechits = 0;
      for (unsigned int i=0;i<rechits.size();++i) {
	const HGCRecHit& hgrh = rechits[i];
	const DET detid(hgrh.detid());
	
	if( subdet != detid.subdetId() ) {
	  throw cms::Exception("IncorrectHGCSubdetector")
	    << "subdet expected: " << subdet 
	    << " subdet gotten: " << detid.subdetId() << std::endl;
	}
	
	double energy = hgrh.energy();
	double time = hgrh.time();	
	
	const CaloCellGeometry *thisCell = geom->getSubdetectorGeometry(detid.det(),detid.subdetId())->getGeometry(detid);
	
	// find rechit geometry
	if(!thisCell) {
	  LogDebug("PFHGCalRecHitCreator")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  ++skipped_rechits;
	  continue;
	}
  

	reco::PFRecHit rh(thisCell, detid.rawId(),Layer,
			   energy); 
	
	//  rh.setOriginalRecHit(edm::Ref<HGCRecHitCollection>(recHitHandle,i));

	
	bool rcleaned = false;
	bool keep=true;

	//Apply Q tests
	for (unsigned int i=0;i<qualityTests_.size();++i) {
	  if (!qualityTests_.at(i)->test(rh,hgrh,rcleaned)) {
	    keep = false;	    
	  }
	}
	  
	if(keep) {
	  rh.setTime(time);
	  out->push_back(rh);
	}
	else if (rcleaned) 
	  cleaned->push_back(rh);
      }
      edm::LogInfo("HGCalRecHitCreator") 
	<<  "Skipped " << skipped_rechits 
	<< " out of " << rechits.size() << " rechits!" << std::endl;
      edm::LogInfo("HGCalRecHitCreator")
	<< "Created " << out->size() << " PFRecHits!" << std::endl;
    }



 protected:
  edm::EDGetTokenT<HGCRecHitCollection> recHitToken_;
  std::string geometryInstance_;

};

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"

typedef PFHGCalRecHitCreator<HGCalDetId,PFLayer::HGCAL,HGCEE> PFHGCEERecHitCreator;
typedef PFHGCalRecHitCreator<HGCalDetId,PFLayer::HGCAL,HGCHEF> PFHGCHEFRecHitCreator;
typedef PFHGCalRecHitCreator<HcalDetId ,PFLayer::HGCAL,HcalEndcap> PFHGCHEBRecHitCreator;


#endif
