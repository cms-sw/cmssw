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
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CondFormats/CastorObjects/interface/CastorRecoParams.h"
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorChannelStatus.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

    
CastorSimpleReconstructor::CastorSimpleReconstructor(edm::ParameterSet const& conf):
  reco_(conf.getParameter<int>("firstSample"),conf.getParameter<int>("samplesToAdd"),conf.getParameter<bool>("correctForTimeslew"),
	conf.getParameter<bool>("correctForPhaseContainment"),conf.getParameter<double>("correctionPhaseNS")),
  det_(DetId::Hcal),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
  firstSample_(conf.getParameter<int>("firstSample")),
  samplesToAdd_(conf.getParameter<int>("samplesToAdd")),
  tsFromDB_(conf.getParameter<bool>("tsFromDB")),
  setSaturationFlag_(conf.getParameter<bool>("setSaturationFlag")),
  maxADCvalue_(conf.getParameter<int>("maxADCvalue"))
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

void CastorSimpleReconstructor::beginRun(edm::Run&r, edm::EventSetup const & es){

  if (tsFromDB_) {
  	edm::ESHandle<CastorRecoParams> p;
  	es.get<CastorRecoParamsRcd>().get(p);
  	if (!p.isValid()) { 
      		tsFromDB_ = false;
      		edm::LogWarning("CastorSimpleReconstructor") << "Could not handle the CastorRecoParamsRcd correctly, using parameters from cfg file. These parameters could be wrong for this run... please check" << std::endl;
  	} else {
      		paramTS_ = new CastorRecoParams(*p.product());
  	}
  }
  
}
void CastorSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<CastorDbService> conditions;
  eventSetup.get<CastorDbRecord>().get(conditions);
  const CastorQIEShape* shape = conditions->getCastorShape (); // this one is generic
  
  CastorCalibrations calibrations;
  
   edm::ESHandle<CastorChannelQuality> p;
   eventSetup.get<CastorChannelQualityRcd>().get(p);
   CastorChannelQuality* myqual = new CastorChannelQuality(*p.product());
  
  
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
      DetId detcell=(DetId)cell;	  
 const CastorCalibrations& calibrations=conditions->getCastorCalibrations(cell);

	// now check the channelquality of this rechit
	bool ok = true;
	std::vector<DetId> channels = myqual->getAllChannels();
	for (std::vector<DetId>::iterator channel = channels.begin();channel !=  channels.end();channel++) {	
		if (channel->rawId() == detcell.rawId()) {
			const CastorChannelStatus* mydigistatus=myqual->getValues(*channel);
			if (mydigistatus->getValue() == 2989) ok = false; // 2989 = BAD
		}
	}

//conditions->makeCastorCalibration (cell, &calibrations);
      
      if (tsFromDB_) {
	  const CastorRecoParam* param_ts = paramTS_->getValues(detcell.rawId());
          reco_.resetTimeSamples(param_ts->firstSample(),param_ts->samplesToAdd());
          //std::cout << "using CastorRecoParam from DB, reco_ parameters are reset to: firstSample_ = " << param_ts->firstSample() << " samplesToAdd_ = " << 
          //param_ts->samplesToAdd() << std::endl;
      }          
      const CastorQIECoder* channelCoder = conditions->getCastorCoder (cell);
      CastorCoderDb coder (*channelCoder, *shape);
      if (ok) {
	rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	if (setSaturationFlag_) reco_.checkADCSaturation(rec->back(),*i,maxADCvalue_);
      }
    }
    // return result
    e.put(rec);     
//     }
  }
}
