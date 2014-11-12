#ifndef RecoParticleFlow_PFClusterProducer_PFHBHeRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFHBHeRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"

#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
class PFHBHERecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFHBHERecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<edm::SortedCollection<HBHERecHit> >(iConfig.getParameter<edm::InputTag>("src"));
      vertexToken_ = iC.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexSrc"));
      offset_last_ = iConfig.getParameter<std::vector<double> >("offset_last");
      offset_prelast_ = iConfig.getParameter<std::vector<double> >("offset_before_last");
    }

    void importRecHits(std::auto_ptr<reco::PFRecHitCollection>&out,std::auto_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {


      edm::ESHandle<HcalDDDRecConstants> hDRCons;
      iSetup.get<HcalRecNumberingRecord>().get(hDRCons);
      int iEtaHEMax=hDRCons->getEtaRange(1).second;

      for (unsigned int i=0;i<qualityTests_.size();++i) {
	qualityTests_.at(i)->beginEvent(iEvent,iSetup);
      }

      edm::Handle<edm::SortedCollection<HBHERecHit> > recHitHandle;
      edm::Handle<reco::VertexCollection> vertexHandle;

      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
      // get the ecal geometry
      const CaloSubdetectorGeometry *hcalBarrelGeo = 
	geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
      const CaloSubdetectorGeometry *hcalEndcapGeo = 
	geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

      iEvent.getByToken(recHitToken_,recHitHandle);
      iEvent.getByToken(vertexToken_,vertexHandle);

      int vertices = vertexHandle->size();
      for( const auto& erh : *recHitHandle ) {      
	const HcalDetId& detid = (HcalDetId)erh.detid();
	HcalSubdetector esd=(HcalSubdetector)detid.subdetId();
	
	double energy = erh.energy();
	double time = erh.time();
	int depth = detid.depth();

	math::XYZVector position;
	math::XYZVector axis;
	
	const CaloCellGeometry *thisCell=0;
	PFLayer::Layer layer = PFLayer::HCAL_BARREL1;
	switch(esd) {
	case HcalBarrel:
	  thisCell =hcalBarrelGeo->getGeometry(detid); 
	  layer =PFLayer::HCAL_BARREL1;
	  break;

	case HcalEndcap:
	  thisCell =hcalEndcapGeo->getGeometry(detid); 
	  layer =PFLayer::HCAL_ENDCAP;
	  break;
	default:
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



	//Decalibration for side effcet
	if (offset_prelast_.size()>0 && abs(detid.ieta()) ==iEtaHEMax-1) {
	  energy = energy - offset_prelast_[depth-1]*vertices;

	}
	if (offset_last_.size()>0 && abs(detid.ieta()) ==iEtaHEMax) {
	  energy = energy - offset_last_[depth-1]*vertices;
	}
  
	reco::PFRecHit rh( detid.rawId(),layer,
			   energy, 
			   position.x(), position.y(), position.z(), 
			   0,0,0);
	rh.setTime(time); //Mike: This we will use later
	rh.setDepth(depth);
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
    edm::EDGetTokenT<edm::SortedCollection<HBHERecHit> > recHitToken_;
    edm::EDGetTokenT<reco::VertexCollection > vertexToken_;
    std::vector<double> offset_last_;
    std::vector<double> offset_prelast_;

};



#endif
