#include "HcalSimpleReconstructor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <iostream>
    
HcalSimpleReconstructor::HcalSimpleReconstructor(edm::ParameterSet const& conf):
  reco_(conf.getParameter<bool>("correctForTimeslew"),
	conf.getParameter<bool>("correctForPhaseContainment"),conf.getParameter<double>("correctionPhaseNS")),
  det_(DetId::Hcal),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
  dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")),
  firstSample_(conf.getParameter<int>("firstSample")),
  samplesToAdd_(conf.getParameter<int>("samplesToAdd")),
  tsFromDB_(conf.getParameter<bool>("tsFromDB"))
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

HcalSimpleReconstructor::~HcalSimpleReconstructor() { }

void HcalSimpleReconstructor::beginRun(edm::Run&r, edm::EventSetup const & es){
  if(tsFromDB_) {
    edm::ESHandle<HcalRecoParams> p;
    es.get<HcalRecoParamsRcd>().get(p);
    paramTS = new HcalRecoParams(*p.product());
  }
  reco_.beginRun(es);
}

void HcalSimpleReconstructor::endRun(edm::Run&r, edm::EventSetup const & es){
  if(tsFromDB_ && paramTS) {
    delete paramTS;
    paramTS = 0;
    reco_.endRun();
  }
}


template<class DIGICOLL, class RECHITCOLL> 
void HcalSimpleReconstructor::process(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);

  // HACK related to HB- corrections
  if(e.isRealData()) reco_.setForData();

  edm::Handle<DIGICOLL> digi;

  e.getByLabel(inputLabel_,digi);

  // create empty output
  std::auto_ptr<RECHITCOLL> rec(new RECHITCOLL);
  rec->reserve(digi->size());
  // run the algorithm
  int first = firstSample_;
  int toadd = samplesToAdd_;
  typename DIGICOLL::const_iterator i;
  for (i=digi->begin(); i!=digi->end(); i++) {
    HcalDetId cell = i->id();
    DetId detcell=(DetId)cell;
    // rof 27.03.09: drop ZS marked and passed digis:
    if (dropZSmarkedPassed_)
      if (i->zsMarkAndPass()) continue;

    const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
    const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
    const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); 
    HcalCoderDb coder (*channelCoder, *shape);

    //>>> firstSample & samplesToAdd
    if(tsFromDB_) {
      const HcalRecoParam* param_ts = paramTS->getValues(detcell.rawId());
      first = param_ts->firstSample();
      toadd = param_ts->samplesToAdd();
    }
    rec->push_back(reco_.reconstruct(*i,first,toadd,coder,calibrations));

  }
  // return result
  e.put(rec);
}


void HcalSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // HACK related to HB- corrections
  if(e.isRealData()) reco_.setForData();
 
  
  if (det_==DetId::Hcal) {
    if (subdet_==HcalBarrel || subdet_==HcalEndcap) {
      process<HBHEDigiCollection, HBHERecHitCollection>(e, eventSetup);
    } else if (subdet_==HcalForward) {
      process<HFDigiCollection, HFRecHitCollection>(e, eventSetup);
    } else if (subdet_==HcalOuter) {
      process<HODigiCollection, HORecHitCollection>(e, eventSetup);
    } else if (subdet_==HcalOther && subdetOther_==HcalCalibration) {
      process<HcalCalibDigiCollection, HcalCalibRecHitCollection>(e, eventSetup);
    }
  } 
}
