#ifndef RecoParticleFlow_PFClusterProducer_PFEcalRecHitCreatorGeomHack_h
#define RecoParticleFlow_PFClusterProducer_PFEcalRecHitCreatorGeomHack_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"


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

template <typename Geometry,PFLayer::Layer Layer,int Detector, typename GeometryRcd>
  class PFEcalRecHitCreatorGeomHack :  public  PFRecHitCreatorBase {

 public:  
  PFEcalRecHitCreatorGeomHack(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
    }

    void importRecHits(std::auto_ptr<reco::PFRecHitCollection>&out,std::auto_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {

      for (unsigned int i=0;i<qualityTests_.size();++i) {
	qualityTests_.at(i)->beginEvent(iEvent,iSetup);
      }


      edm::Handle<EcalRecHitCollection> recHitHandle;

      edm::ESHandle<Geometry> geoHandle;
      iSetup.get<GeometryRcd>().get(geoHandle);
      
      const Geometry *ecalGeo = geoHandle.product();

      iEvent.getByToken(recHitToken_,recHitHandle);
      for (unsigned int i=0;i<recHitHandle->size();++i) {
	const EcalRecHit& erh = (*recHitHandle)[i];
	const DetId& detid = erh.detid();
	double energy = erh.energy();
	double time = erh.time();

	math::XYZVector position;
	math::XYZVector axis;
	
	const CaloCellGeometry *thisCell;
	thisCell= ecalGeo->getGeometry(detid);
  
	// find rechit geometry
	if(!thisCell) {
	  edm::LogError("PFEcalRecHitCreatorGeomHack")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  continue;
	}
  
	position.SetCoordinates ( thisCell->getPosition().x(),
				  thisCell->getPosition().y(),
				  thisCell->getPosition().z() );
  
	// the axis vector is the difference 
	const TruncatedPyramid* pyr 
	  = dynamic_cast< const TruncatedPyramid* > (thisCell);    


	if( pyr ) {
	  axis.SetCoordinates( pyr->getPosition(1).x(), 
			       pyr->getPosition(1).y(), 
			       pyr->getPosition(1).z() ); 
    
	  math::XYZVector axis0( pyr->getPosition(0).x(), 
				 pyr->getPosition(0).y(), 
				 pyr->getPosition(0).z() );
    
	  axis -= axis0;    
	}
	else continue;

	reco::PFRecHit rh( detid.rawId(),Layer,
			   energy, 
			   position.x(), position.y(), position.z(), 
			   axis.x(), axis.y(), axis.z() ); 


	
	const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();
	assert( corners.size() == 8 );

	rh.setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
	rh.setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
	rh.setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
	rh.setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );
	

	bool rcleaned = false;
	bool keep=true;

	//Apply Q tests
	for (unsigned int i=0;i<qualityTests_.size();++i) {
	  if (!qualityTests_.at(i)->test(rh,erh,rcleaned)) {
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
    }



 protected:
  edm::EDGetTokenT<EcalRecHitCollection> recHitToken_;


};

#include "Geometry/FCalGeometry/interface/ShashlikGeometry.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"
//typedef PFEcalRecHitCreatorGeomHack<ShashlikGeometry,PFLayer::ECAL_ENDCAP,EcalShashlik,ShashlikGeometryRecord> PFEKRecHitCreator;

#endif
