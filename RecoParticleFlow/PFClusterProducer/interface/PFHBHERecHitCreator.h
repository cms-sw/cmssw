#ifndef RecoParticleFlow_PFClusterProducer_PFHcalRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFHcalRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"

#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
//typedef PFHBHERecHitCreator<HBHERecHit,CaloSubdetectorGeometry,PFLayer::HCAL_BARREL1,HcalBarrel> PFHBRecHitCreator;
//template <typename Digi, typename Geometry,PFLayer::Layer Layer,int Detector>
class PFHBHERecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFHBHERecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<edm::SortedCollection<HBHERecHit> >(iConfig.getParameter<edm::InputTag>("src"));
    }

    void importRecHits(std::auto_ptr<reco::PFRecHitCollection>&out,std::auto_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {

      for (unsigned int i=0;i<qualityTests_.size();++i) {
	qualityTests_.at(i)->beginEvent(iEvent,iSetup);
      }

      edm::Handle<edm::SortedCollection<HBHERecHit> > recHitHandle;

      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
      // get the ecal geometry
      const CaloSubdetectorGeometry *hcalBarrelGeo = 
	geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
      const CaloSubdetectorGeometry *hcalEndcapGeo = 
	geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

      iEvent.getByToken(recHitToken_,recHitHandle);
      for( const auto& erh : *recHitHandle ) {      
	const HcalDetId& detid = (HcalDetId)erh.detid();
	HcalSubdetector esd=(HcalSubdetector)detid.subdetId();
	
	double energy = erh.energy();
	double time = erh.time();


	math::XYZVector position;
	math::XYZVector axis;
	
	const CaloCellGeometry *thisCell;
	PFLayer::Layer layer;
	switch(esd) {
	case HcalBarrel:
	  thisCell =hcalBarrelGeo->getGeometry(detid); 
	  layer =HCAL_BARREL1;
	  break;

	case HcalEndcap:
	  thisCell =hcalEndcapGeo->getGeometry(detid); 
	  layer =HCAL_ENDCAP;
	  break;

	}
  
	// find rechit geometry
	if(!thisCell) {
	  edm::LogError("PFHBHERecHitCreator")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  continue;
	}
  
	position.SetCoordinates ( thisCell->getPosition().x(),
				  thisCell->getPosition().y(),
				  thisCell->getPosition().z() );
  
	reco::PFRecHit rh( detid.rawId(),layer,
			   energy, 
			   position.x(), position.y(), position.z(), 
			   0,0,0);
	rh.setTime(time); //Mike: This we will use later

	const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();
	assert( corners.size() == 8 );

	rh.setNECorner( corners[0].x(), corners[0].y(),  corners[0].z());
	rh.setSECorner( corners[1].x(), corners[1].y(),  corners[1].z());
	rh.setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z());
	rh.setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z());
	

	bool rcleaned = false;
	bool keep=true;

	//Apply Q tests
	for( const auto& qtest : qualityTests_ ) {
	  if (!qtest->test(rh,erh,rcleaned)) {
	    keep = false;
	    
	  }
	}
	  
	if(keep) {
	  out->push_back(rh);
	}
	else if (rcleaned) 
	  cleaned->push_back(rh);
      }
    }



 protected:
    edm::EDGetTokenT<edm::SortedCollection<Digi> > recHitToken_;


};



#endif
