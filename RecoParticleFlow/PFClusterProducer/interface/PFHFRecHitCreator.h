#ifndef RecoParticleFlow_PFClusterProducer_PFHFRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFHFRecHitCreator_h

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

class PFHFRecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFHFRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<edm::SortedCollection<HFRecHit>  >(iConfig.getParameter<edm::InputTag>("src"));
      EM_Depth_ = iConfig.getParameter<double>("EMDepthCorrection");
      HAD_Depth_ = iConfig.getParameter<double>("HADDepthCorrection");
      shortFibre_Cut = iConfig.getParameter<double>("ShortFibre_Cut");
      longFibre_Fraction = iConfig.getParameter<double>("LongFibre_Fraction");
      longFibre_Cut = iConfig.getParameter<double>("LongFibre_Cut");
      shortFibre_Fraction = iConfig.getParameter<double>("ShortFibre_Fraction");
      thresh_HF_ =    iConfig.getParameter<double>("thresh_HF");
      HFCalib_ =    iConfig.getParameter<double>("HFCalib29");

    }



    void importRecHits(std::auto_ptr<reco::PFRecHitCollection>&out,std::auto_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {


      reco::PFRecHitCollection tmpOut;

      beginEvent(iEvent,iSetup);

      edm::Handle<edm::SortedCollection<HFRecHit> > recHitHandle;

      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
      // get the ecal geometry
      const CaloSubdetectorGeometry *hcalGeo = 
	geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalForward);

      iEvent.getByToken(recHitToken_,recHitHandle);
      for( const auto& erh : *recHitHandle ) {      
	const HcalDetId& detid = (HcalDetId)erh.detid();
	int depth = detid.depth();

	double energy = erh.energy();
	double time = erh.time();

	math::XYZVector position;
	math::XYZVector axis;
	
	const CaloCellGeometry *thisCell;
	thisCell= hcalGeo->getGeometry(detid);

	// find rechit geometry
	if(!thisCell) {
	  edm::LogError("PFHFRecHitCreator")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  continue;
	}

	auto const point =  thisCell->getPosition();


	PFLayer::Layer layer;
	double depth_correction;
	if (depth==1) {
	  layer = PFLayer::HF_EM;
          depth_correction = point.z() > 0. ? EM_Depth_ : -EM_Depth_;
	}
	else {
	  layer = PFLayer::HF_HAD;
	  depth_correction = point.z() > 0. ? HAD_Depth_ : -HAD_Depth_;
	}

  
	position.SetCoordinates ( point.x(),
				  point.y(),
				  point.z()+depth_correction );


	reco::PFRecHit rh( detid.rawId(),layer,
			   energy, 
			   position.x(), position.y(), position.z(), 
			   0,0,0);
	rh.setTime(time); 
	rh.setDepth(depth);

	const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();
	assert( corners.size() == 8 );

	rh.setNECorner( corners[0].x(), corners[0].y(),  corners[0].z()+depth_correction);
	rh.setSECorner( corners[1].x(), corners[1].y(),  corners[1].z()+depth_correction);
	rh.setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z()+depth_correction);
	rh.setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z()+depth_correction);
	

	bool rcleaned = false;
	bool keep=true;

	//Apply Q tests
	for( const auto& qtest : qualityTests_ ) {
	  if (!qtest->test(rh,erh,rcleaned)) {
	    keep = false;
	    
	  }
	}
	  
	if(keep) {
	  tmpOut.push_back(rh);
	}
	else if (rcleaned) 
	  cleaned->push_back(rh);
      }
      //Sort by DetID the collection
      DetIDSorter sorter;
      if (tmpOut.size()>0)
	std::sort(tmpOut.begin(),tmpOut.end(),sorter); 


      /////////////////////HF DUAL READOUT/////////////////////////
      
      double lONG=0.;
      double sHORT=0.;

      for (auto& hit : tmpOut) {
	lONG=0.0;
	sHORT=0.0;

	reco::PFRecHit newHit = hit;
	const HcalDetId& detid = (HcalDetId)hit.detId();
	if (detid.depth()==1) {
	  lONG=hit.energy();
	  //find the short hit
	  HcalDetId shortID (HcalForward, detid.ieta(), detid.iphi(), 2);
	  const reco::PFRecHit temp(shortID,PFLayer::NONE,0.0,math::XYZPoint(0,0,0),math::XYZVector(0,0,0),std::vector<math::XYZPoint>());
	  auto found_hit = std::lower_bound(tmpOut.begin(),tmpOut.end(),
					    temp,
					    [](const reco::PFRecHit& a, 
					       const reco::PFRecHit& b){
					      return a.detId() < b.detId();
					    });
	  if( found_hit != tmpOut.end() && found_hit->detId() == shortID.rawId() ) {
	    sHORT = found_hit->energy();
	    //Ask for fraction
	    double energy = lONG-sHORT;

	    if (abs(detid.ieta())<=32)
	      energy*=HFCalib_;
	    newHit.setEnergy(energy);
	    if (!( lONG > longFibre_Cut && 
		   ( sHORT/lONG < shortFibre_Fraction)))
	      if (energy>thresh_HF_)
		out->push_back(newHit);
	  }
	  else
	    {
	      //make only long hit
	      double energy = lONG;
	      if (abs(detid.ieta())<=32)
		energy*=HFCalib_;
	      newHit.setEnergy(energy);

	      if (energy>thresh_HF_)
		out->push_back(newHit);

	    }

	}
	else {
	  sHORT=hit.energy();
	  HcalDetId longID (HcalForward, detid.ieta(), detid.iphi(), 1);
	  const reco::PFRecHit temp(longID,PFLayer::NONE,0.0,math::XYZPoint(0,0,0),math::XYZVector(0,0,0),std::vector<math::XYZPoint>());
	  auto found_hit = std::lower_bound(tmpOut.begin(),tmpOut.end(),
					    temp,
					    [](const reco::PFRecHit& a, 
					       const reco::PFRecHit& b){
					      return a.detId() < b.detId();
					    });
	  double energy = 2*sHORT;
	  if( found_hit != tmpOut.end() && found_hit->detId() == longID.rawId() ) {
	    lONG = found_hit->energy();
	    //Ask for fraction

	    //If in this case lONG-sHORT<0 add the energy to the sHORT
	    if ((lONG-sHORT)<thresh_HF_)
	      energy = lONG+sHORT;

	    if (abs(detid.ieta())<=32)
	      energy*=HFCalib_;
	    
	    newHit.setEnergy(energy);
	    if (!( sHORT > shortFibre_Cut && 
		   ( lONG/sHORT < longFibre_Fraction)))
	      if (energy>thresh_HF_)
		out->push_back(newHit);

	  }
	  else {
	    //only short hit!
	    if (abs(detid.ieta())<=32)
	      energy*=HFCalib_;
	    newHit.setEnergy(energy);
	      if (energy>thresh_HF_)
		out->push_back(newHit);
	  }
	}


      }

    }



 protected:
    edm::EDGetTokenT<edm::SortedCollection<HFRecHit> > recHitToken_;
    double EM_Depth_;
    double HAD_Depth_;
    // Don't allow large energy in short fibres if there is no energy in long fibres
    double shortFibre_Cut;  
    double longFibre_Fraction;

    // Don't allow large energy in long fibres if there is no energy in short fibres
    double longFibre_Cut;  
    double shortFibre_Fraction;
    double           thresh_HF_;
    double HFCalib_;

    class DetIDSorter {
    public:
      DetIDSorter() {}
      ~DetIDSorter() {}

      bool operator()(const reco::PFRecHit& a,
		     const reco::PFRecHit& b) {
      return a.detId() < b.detId();
    }

    };


};
#endif
