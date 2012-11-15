using namespace std;
#include "CastorSimpleReconstructor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
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
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"
#include "CondFormats/CastorObjects/interface/CastorSaturationCorrs.h"

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
  maxADCvalue_(conf.getParameter<int>("maxADCvalue")),
  doSaturationCorr_(conf.getParameter<bool>("doSaturationCorr"))
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
  
  // read out saturation correction factors for this run
  edm::ESHandle<CastorSaturationCorrs> satcorr;
  es.get<CastorSaturationCorrsRcd>().get(satcorr);
  if (!satcorr.isValid()) { 
      	edm::LogWarning("CastorSimpleReconstructor") << "Could not handle the CastorSaturationCorrsRcd correctly." << std::endl;
  } else {
      	satCorr_ = new CastorSaturationCorrs(*satcorr.product());
  }
  
}
void CastorSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<CastorDbService> conditions;
  eventSetup.get<CastorDbRecord>().get(conditions);
  const CastorQIEShape* shape = conditions->getCastorShape (); // this one is generic
  
  CastorCalibrations calibrations;

  
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

      
      if (tsFromDB_) {
	  const CastorRecoParam* param_ts = paramTS_->getValues(detcell.rawId());
          reco_.resetTimeSamples(param_ts->firstSample(),param_ts->samplesToAdd());
          //std::cout << "using CastorRecoParam from DB, reco_ parameters are reset to: firstSample_ = " << param_ts->firstSample() << " samplesToAdd_ = " << 
          //param_ts->samplesToAdd() << std::endl;
      }          
      const CastorQIECoder* channelCoder = conditions->getCastorCoder (cell);
      CastorCoderDb coder (*channelCoder, *shape);
      
      // get saturation correction value
      const CastorSaturationCorr* saturationCorr = satCorr_->getValues(detcell.rawId());
      double satCorrConst = 1.;
      satCorrConst = saturationCorr->getValue();
      //std::cout << " satCorrConst = " << satCorrConst << std::endl;
      
      rec->push_back(reco_.reconstruct(*i,coder,calibrations));
      if (setSaturationFlag_) reco_.checkADCSaturation(rec->back(),*i,maxADCvalue_);

      //++++ Saturation Correction +++++
      // only for a real event try to recover the energy
      if (e.eventAuxiliary().isRealData() && doSaturationCorr_) 
	reco_.recoverADCSaturation(rec->back(),coder,calibrations,*i,maxADCvalue_,satCorrConst);
    }
    // return result
    e.put(rec);     
  }
}
