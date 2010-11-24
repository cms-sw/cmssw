using namespace std;
#include "CastorSimpleReconstructor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

    
CastorSimpleReconstructor::CastorSimpleReconstructor(edm::ParameterSet const& conf):
  reco_(conf.getParameter<int>("firstSample"),conf.getParameter<int>("samplesToAdd"),conf.getParameter<bool>("correctForTimeslew"),
	conf.getParameter<bool>("correctForPhaseContainment"),conf.getParameter<double>("correctionPhaseNS")),
  det_(DetId::Hcal),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel"))	
{
  std::string subd=conf.getParameter<std::string>("Subdetector");
  if (!strcasecmp(subd.c_str(),"CASTOR")) {
    det_=DetId::Calo;
    subdet_=HcalCastorDetId::SubdetectorId;
    produces<CastorRecHitCollection>();
  } else {
    edm::LogWarning("CastorSimpleReconstructor") << "CastorSimpleReconstructor is not associated with CASTOR subdetector!" << std::endl;
  }       
  
}

CastorSimpleReconstructor::~CastorSimpleReconstructor() {
}

void CastorSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<CastorDbService> conditions;
  eventSetup.get<CastorDbRecord>().get(conditions);
  const CastorQIEShape* shape = conditions->getCastorShape (); // this one is generic
  
  // some hard coding for 2009 data
  // get event number 
  int runNumber = e.run();
  if (runNumber < 129456) { CastorSimpleRecAlgo usedRecAlgo2009(1,4,false,false,0.0); reco_ = usedRecAlgo2009;}
  
  CastorCalibrations calibrations;
  
//  if (det_==DetId::Hcal) {
     if (det_==DetId::Calo && subdet_==HcalCastorDetId::SubdetectorId) {
    edm::Handle<CastorDigiCollection> digi;
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<CastorRecHitCollection> rec(new CastorRecHitCollection);
    // run the algorithm
    CastorDigiCollection::const_iterator i;
    for (i=digi->begin(); i!=digi->end(); i++) {
      HcalCastorDetId cell = i->id();	  
 const CastorCalibrations& calibrations=conditions->getCastorCalibrations(cell);


//conditions->makeCastorCalibration (cell, &calibrations);

      const CastorQIECoder* channelCoder = conditions->getCastorCoder (cell);
      CastorCoderDb coder (*channelCoder, *shape);
      rec->push_back(reco_.reconstruct(*i,coder,calibrations));
    }
    // return result
    e.put(rec);     
//     }
  }
}
