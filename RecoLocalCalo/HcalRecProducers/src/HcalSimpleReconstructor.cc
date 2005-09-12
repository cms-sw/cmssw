using namespace std;
#include "RecoLocalCalo/HcalRecProducers/interface/HcalSimpleReconstructor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/EDProduct/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbServiceHandle.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <iostream>

namespace cms
{
  
  namespace hcal 
  { 
    class HRSHappySelector : public edm::Selector {
    public:
      HRSHappySelector() { }
    private:
      virtual bool doMatch(const edm::Provenance& p) const {
	//      cout << p << endl;
	return true;
      }
    };
    
    
    HcalSimpleReconstructor::HcalSimpleReconstructor(edm::ParameterSet const& conf):
      reco_(conf.getParameter<int>("firstSample"),conf.getParameter<int>("samplesToAdd"))
	
    {
      std::string subd=conf.getParameter<std::string>("Subdetector");
      if (!strcasecmp(subd.c_str(),"HBHE")) {
	subdet_=HcalBarrel;
	produces<HBHERecHitCollection>();
      } else if (!strcasecmp(subd.c_str(),"HO")) {
	subdet_=HcalOuter;
	produces<HORecHitCollection>();
      } else if (!strcasecmp(subd.c_str(),"HF")) {
	subdet_=HcalForward;
	produces<HFRecHitCollection>();
      } else {
	std::cout << "HcalSimpleReconstructor is not associated with a specific subdetector!" << std::endl;
      }       
      
    }
    
    HcalSimpleReconstructor::~HcalSimpleReconstructor() {
    }
    
    void HcalSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
    {
      // get calibrations
      edm::eventsetup::ESHandle<HcalDbServiceHandle> pSetup;
      eventSetup.get<HcalDbRecord>().get(pSetup);
      const QieShape* qieShape = pSetup->getBasicShape (); // this one is generic
      
      if (subdet_==HcalBarrel || subdet_==HcalEndcap) {
	edm::Handle<cms::HBHEDigiCollection> digi;
	// selector?
	HRSHappySelector s;
	e.get(s, digi);
	
	// create empty output
	std::auto_ptr<cms::HBHERecHitCollection> rec(new cms::HBHERecHitCollection);
	// run the algorithm
	cms::HBHEDigiCollection::const_iterator i;
	for (i=digi->begin(); i!=digi->end(); i++) {
	  cms::HcalDetId cell = i->id();
	  const HcalCalibrations* calibrations = pSetup->getHcalCalibrations (cell);
	  const HcalChannelCoder* channelCoder = pSetup->getChannelCoder (cell);
	  HcalCoderDb coder (*channelCoder, *qieShape);
	  rec->push_back(reco_.reconstruct(*i,coder,*calibrations));
	}
	// return result
	e.put(rec);
      } else if (subdet_==HcalOuter) {
	edm::Handle<cms::HODigiCollection> digi;
	// selector?
	HRSHappySelector s;
	e.get(s, digi);
	
	// create empty output
	std::auto_ptr<cms::HORecHitCollection> rec(new cms::HORecHitCollection);
	// run the algorithm
	cms::HODigiCollection::const_iterator i;
	for (i=digi->begin(); i!=digi->end(); i++) {
	  cms::HcalDetId cell = i->id();
	  const HcalCalibrations* calibrations = pSetup->getHcalCalibrations (cell);
	  const HcalChannelCoder* channelCoder = pSetup->getChannelCoder (cell);
	  HcalCoderDb coder (*channelCoder, *qieShape);
	  rec->push_back(reco_.reconstruct(*i,coder,*calibrations));
	}
	// return result
	e.put(rec);    
      } else if (subdet_==HcalForward) {
	edm::Handle<cms::HFDigiCollection> digi;
	// selector?
	HRSHappySelector s;
	e.get(s, digi);
	
	// create empty output
	std::auto_ptr<cms::HFRecHitCollection> rec(new cms::HFRecHitCollection);
	// run the algorithm
	cms::HFDigiCollection::const_iterator i;
	for (i=digi->begin(); i!=digi->end(); i++) {
	  cms::HcalDetId cell = i->id();
	  std::cout << "HcalSimpleReconstructor::produce-> HF ID: " << cell << std::endl;
	  const HcalCalibrations* calibrations = pSetup->getHcalCalibrations (cell);
	  const HcalChannelCoder* channelCoder = pSetup->getChannelCoder (cell);
	  HcalCoderDb coder (*channelCoder, *qieShape);
	  rec->push_back(reco_.reconstruct(*i,coder,*calibrations));
	}
	// return result
	e.put(rec);     
      }
    }
  }
}
