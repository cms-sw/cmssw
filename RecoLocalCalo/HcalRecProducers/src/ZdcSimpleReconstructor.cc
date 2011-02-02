using namespace std;
#include "ZdcSimpleReconstructor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <iostream>

    
ZdcSimpleReconstructor::ZdcSimpleReconstructor(edm::ParameterSet const& conf):
  reco_(conf.getParameter<int>("firstSample"),conf.getParameter<int>("firstNoise"),conf.getParameter<int>("samplesToAdd"),conf.getParameter<bool>("correctForTimeslew"),
	conf.getParameter<bool>("correctForPhaseContainment"),conf.getParameter<double>("correctionPhaseNS"),
	conf.getParameter<int>("recoMethod")),
  det_(DetId::Hcal),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
  dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed"))
{
  std::string subd=conf.getParameter<std::string>("Subdetector");
  if (!strcasecmp(subd.c_str(),"ZDC")) {
    det_=DetId::Calo;
    subdet_=HcalZDCDetId::SubdetectorId;
    produces<ZDCRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"CALIB")) {
    subdet_=HcalOther;
    subdetOther_=HcalCalibration;
    produces<HcalCalibRecHitCollection>();
  } else {
    std::cout << "ZdcSimpleReconstructor is not associated with a specific subdetector!" << std::endl;
  }       
  
}

ZdcSimpleReconstructor::~ZdcSimpleReconstructor() {
}

void ZdcSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  const HcalQIEShape* shape = conditions->getHcalShape (); // this one is generic
  
  if (det_==DetId::Calo && subdet_==HcalZDCDetId::SubdetectorId) {
    edm::Handle<ZDCDigiCollection> digi;
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<ZDCRecHitCollection> rec(new ZDCRecHitCollection);
    rec->reserve(digi->size());
    // run the algorithm
    ZDCDigiCollection::const_iterator i;
    for (i=digi->begin(); i!=digi->end(); i++) {
      HcalZDCDetId cell = i->id();	  
	// rof 27.03.09: drop ZS marked and passed digis:
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

      const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
      const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
      HcalCoderDb coder (*channelCoder, *shape);
      rec->push_back(reco_.reconstruct(*i,coder,calibrations));
    }
    // return result
    e.put(rec);     
  }
}
