#ifndef RecoParticleFlow_PFClusterProducer_PFHBHeRecHitCreatorPulses_h
#define RecoParticleFlow_PFClusterProducer_PFHBHeRecHitCreatorPulses_h

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

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"


#include "CalibFormats/HcalObjects/interface/HcalDbService.h"


#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
class PFHBHERecHitCreatorPulses :  public  PFRecHitCreatorBase {

 public:  
  PFHBHERecHitCreatorPulses(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
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


      edm::ESHandle<HcalDbService> conditions;
      iSetup.get<HcalDbRecord > ().get(conditions);
  
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


	//CUSTOM ENERGY AND TIME RECO
	CaloSamples tool;
	HcalCalibrations calibrations = conditions->getHcalCalibrations(detid);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder(detid);
	const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
	HcalCoderDb coder(*channelCoder, *shape);

	int auxwd1 = erh.auxHBHE();  // TS = 0,1,2,3 info
	int auxwd2 = erh.aux();      // TS = 4,5,6,7 info 	

	
	int adc[8];
	int capid[8];

	adc[0] = (auxwd1)       & 0x7F;
	adc[1] = (auxwd1 >> 7)  & 0x7F;
	adc[2] = (auxwd1 >> 14) & 0x7F;
	adc[3] = (auxwd1 >> 21) & 0x7F;
	
	capid[0] = (auxwd1 >> 28) & 0x3;  // rotating through 4 values: 0,1,2,3
	capid[1] = (capid[0] + 1 <= 3) ? capid[0] + 1 : 0;
	capid[2] = (capid[1] + 1 <= 3) ? capid[1] + 1 : 0;
	capid[3] = (capid[2] + 1 <= 3) ? capid[2] + 1 : 0;

	adc[4] = (auxwd2)       & 0x7F;
	adc[5] = (auxwd2 >> 7)  & 0x7F;
	adc[6] = (auxwd2 >> 14) & 0x7F;
	adc[7] = (auxwd2 >> 21) & 0x7F;

	capid[4] = (auxwd2 >> 28) & 0x3;
	capid[5] = (capid[4] + 1 <= 3) ? capid[4] + 1 : 0;
	capid[6] = (capid[5] + 1 <= 3) ? capid[5] + 1 : 0;
	capid[7] = (capid[6] + 1 <= 3) ? capid[6] + 1 : 0; 

	// Pack the above into HBHEDataFrame for subsequent linearization to fC 	
	HBHEDataFrame digi(detid);
	digi.setSize(10);
	digi.setPresamples(4); 
	for (int ii = 0; ii < 8; ii++) {
	  HcalQIESample s (adc[ii], capid[ii], 0, 0);
	  digi.setSample(ii,s);
	} 
	

	// Convert/linearize ADC to fC 
	coder.adc2fC(digi, tool); 


	// Get gain (GeV/fC)

	HcalGenericDetId hcalGenDetId(erh.id());
	//	const HcalPedestal* pedestal = conditions->getPedestal(hcalGenDetId);
	//	const HcalGain* gain = conditions->getGain(hcalGenDetId);

	// Convert all 8 TS from fC to GeV (also subtractibg pedestal)

	std::vector<double> samples;

	for (int ii = 0; ii < 8; ii++) {
	  samples.push_back(calibrations.respcorrgain(capid[ii]) *
			    (tool[ii] - calibrations.pedestal(capid[ii]))); 

	  printf("SAMPLE %d ,%f\n",ii,calibrations.respcorrgain(capid[ii]) *
		 (tool[ii] - calibrations.pedestal(capid[ii])));
	} 

	/////////////////////////////
	/////////////////////////////

	//NAIVE ALGO By michalis -> Find the maximum and assign the maximum energy
	//	size_t maxSample  =  (std::max_element(samples.begin(),samples.end()))-samples.begin();


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
	  edm::LogError("PFHBHERecHitCreatorPulses")
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


};



#endif
