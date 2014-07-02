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

class PFPSRecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFPSRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
    }

    void importRecHits(std::auto_ptr<reco::PFRecHitCollection>&out,std::auto_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {

      for (unsigned int i=0;i<qualityTests_.size();++i) {
	qualityTests_.at(i)->beginEvent(iEvent,iSetup);
      }


      edm::Handle<EcalRecHitCollection> recHitHandle;
      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
      // get the ecal geometry
      const CaloSubdetectorGeometry *psGeometry = 
	geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

      iEvent.getByToken(recHitToken_,recHitHandle);
      for( const auto& erh : *recHitHandle ) {      
	ESDetId detid(erh.detid());
	double energy = erh.energy();


	math::XYZVector position;

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
 

	
	const CaloCellGeometry *thisCell;
	thisCell= psGeometry->getGeometry(detid);
  
	// find rechit geometry
	if(!thisCell) {
	  edm::LogError("PFPSRecHitCreator")
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
			   0.0,0.0,0.0);


	
	const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();
	assert( corners.size() == 8 );

	rh.setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
	rh.setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
	rh.setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
	rh.setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );
	

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
  edm::EDGetTokenT<EcalRecHitCollection> recHitToken_;


};


#endif
