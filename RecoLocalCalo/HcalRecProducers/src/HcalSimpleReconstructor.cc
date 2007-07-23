using namespace std;
#include "RecoLocalCalo/HcalRecProducers/interface/HcalSimpleReconstructor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <iostream>

    
    HcalSimpleReconstructor::HcalSimpleReconstructor(edm::ParameterSet const& conf):
      reco_(conf.getParameter<int>("firstSample"),conf.getParameter<int>("samplesToAdd"),conf.getParameter<bool>("correctForTimeslew"),
	    conf.getParameter<bool>("correctForPhaseContainment"),conf.getParameter<double>("correctionPhaseNS")),
      inputLabel_(conf.getParameter<edm::InputTag>("digiLabel"))	
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
      } else if (!strcasecmp(subd.c_str(),"ZDC")) {
	subdet_=HcalOther;
	subdetOther_=HcalZDC;
	produces<ZDCRecHitCollection>();
      } else if (!strcasecmp(subd.c_str(),"CALIB")) {
	subdet_=HcalOther;
	subdetOther_=HcalCalibration;
	produces<HcalCalibRecHitCollection>();
      } else {
	std::cout << "HcalSimpleReconstructor is not associated with a specific subdetector!" << std::endl;
      }       
      
    }
    
    HcalSimpleReconstructor::~HcalSimpleReconstructor() {
    }
    
    void HcalSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
    {
      // get conditions
      edm::ESHandle<HcalDbService> conditions;
      eventSetup.get<HcalDbRecord>().get(conditions);
      const HcalQIEShape* shape = conditions->getHcalShape (); // this one is generic

      HcalCalibrations calibrations;
      
      if (subdet_==HcalBarrel || subdet_==HcalEndcap) {
	edm::Handle<HBHEDigiCollection> digi;
		
	e.getByLabel(inputLabel_,digi);
		
	// create empty output
	std::auto_ptr<HBHERecHitCollection> rec(new HBHERecHitCollection);
	// run the algorithm
	HBHEDigiCollection::const_iterator i;
	for (i=digi->begin(); i!=digi->end(); i++) {
	  HcalDetId cell = i->id();
	  conditions->makeHcalCalibration (cell, &calibrations);
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  HcalCoderDb coder (*channelCoder, *shape);
	  rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	}
	// return result
	e.put(rec);
      } else if (subdet_==HcalOuter) {
	edm::Handle<HODigiCollection> digi;
	e.getByLabel(inputLabel_,digi);
	
	// create empty output
	std::auto_ptr<HORecHitCollection> rec(new HORecHitCollection);
	// run the algorithm
	HODigiCollection::const_iterator i;
	for (i=digi->begin(); i!=digi->end(); i++) {
	  HcalDetId cell = i->id();
	  conditions->makeHcalCalibration (cell, &calibrations);
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  HcalCoderDb coder (*channelCoder, *shape);
	  rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	}
	// return result
	e.put(rec);    
      } else if (subdet_==HcalForward) {
	edm::Handle<HFDigiCollection> digi;
	e.getByLabel(inputLabel_,digi);
	
	// create empty output
	std::auto_ptr<HFRecHitCollection> rec(new HFRecHitCollection);
	// run the algorithm
	HFDigiCollection::const_iterator i;
	for (i=digi->begin(); i!=digi->end(); i++) {
	  HcalDetId cell = i->id();	  
	  conditions->makeHcalCalibration (cell, &calibrations);
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  HcalCoderDb coder (*channelCoder, *shape);
	  rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	}
	// return result
	e.put(rec);     
      } else if (subdet_==HcalOther && subdetOther_==HcalZDC) {
	edm::Handle<ZDCDigiCollection> digi;
	e.getByLabel(inputLabel_,digi);
	
	// create empty output
	std::auto_ptr<ZDCRecHitCollection> rec(new ZDCRecHitCollection);
	// run the algorithm
	ZDCDigiCollection::const_iterator i;
	for (i=digi->begin(); i!=digi->end(); i++) {
	  HcalZDCDetId cell = i->id();	  
	  conditions->makeHcalCalibration (cell, &calibrations);
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  HcalCoderDb coder (*channelCoder, *shape);
	  rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	}
	// return result
	e.put(rec);     
      } else if (subdet_==HcalOther && subdetOther_==HcalCalibration) {
	edm::Handle<HcalCalibDigiCollection> digi;
	e.getByLabel(inputLabel_,digi);
	
	// create empty output
	std::auto_ptr<HcalCalibRecHitCollection> rec(new HcalCalibRecHitCollection);
	// run the algorithm
	HcalCalibDigiCollection::const_iterator i;
	for (i=digi->begin(); i!=digi->end(); i++) {
	  HcalCalibDetId cell = i->id();	  
	  conditions->makeHcalCalibration (cell, &calibrations);
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  HcalCoderDb coder (*channelCoder, *shape);
	  rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	}
	// return result
	e.put(rec);     
      }
    }
