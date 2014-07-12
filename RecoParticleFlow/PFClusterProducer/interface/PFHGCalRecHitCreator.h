#ifndef RecoParticleFlow_PFClusterProducer_PFHGCalRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFHGCalRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"

#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"
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

template <typename DET,PFLayer::Layer Layer,ForwardSubdetector subdet>
  class PFHGCalRecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFHGCalRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
      geometryInstance_ = iConfig.getParameter<std::string>("geometryInstance");
    }

    void importRecHits(std::auto_ptr<reco::PFRecHitCollection>&out,std::auto_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {

      for (unsigned int i=0;i<qualityTests_.size();++i) {
	qualityTests_.at(i)->beginEvent(iEvent,iSetup);
      }

      edm::Handle<HGCRecHitCollection> recHitHandle;
      iEvent.getByToken(recHitToken_,recHitHandle);
      const HGCRecHitCollection& rechits = *recHitHandle;

      edm::ESHandle<HGCalGeometry> geoHandle;
      iSetup.get<IdealGeometryRecord>().get(geometryInstance_,geoHandle);
      const HGCalGeometry& hgcGeo = *geoHandle;
      
      unsigned skipped_rechits = 0;
      for (unsigned int i=0;i<rechits.size();++i) {
	const HGCRecHit& hgrh = rechits[i];
	const DET detid(hgrh.detid());
	
	if( subdet != detid.subdet() ) {
	  throw cms::Exception("IncorrectHGCSubdetector")
	    << "subdet expected: " << subdet 
	    << " subdet gotten: " << detid.subdet() << std::endl;
	}
	
	double energy = hgrh.energy();
	double time = hgrh.time();	
	
	const FlatTrd *thisCell = 
	  static_cast<const FlatTrd*>(hgcGeo.getGeometry(detid));	

	// find rechit geometry
	if(!thisCell) {
	  LogDebug("PFHGCalRecHitCreator")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  ++skipped_rechits;
	  continue;
	}
  
	const GlobalPoint position( std::move( hgcGeo.getPosition( detid ) ) );
	//std::cout << "geometry cell position: " << position << std::endl;

	reco::PFRecHit rh( detid.rawId(),Layer,
			   energy, 
			   position.x(), position.y(), position.z(), 
			   0, 0, 0 ); 
	
	rh.setOriginalRecHit(edm::Ref<HGCRecHitCollection>(recHitHandle,i));

	const HGCalGeometry::CornersVec corners( std::move( hgcGeo.getCorners( detid ) ) );
	assert( corners.size() == 8 );

	rh.setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
	rh.setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
	rh.setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
	rh.setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );
	
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

typedef PFHGCalRecHitCreator<HGCEEDetId,PFLayer::HGC_ECAL,HGCEE> PFHGCEERecHitCreator;
typedef PFHGCalRecHitCreator<HGCHEDetId,PFLayer::HGC_HCALF,HGCHEF> PFHGCHEFRecHitCreator;
typedef PFHGCalRecHitCreator<HGCHEDetId,PFLayer::HGC_HCALB,HGCHEB> PFHGCHEBRecHitCreator;


#endif
