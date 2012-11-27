#include "ZdcHitReconstructor.h"
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
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

#include <iostream>

/*  Zdc Hit reconstructor allows for CaloRecHits with status words */

ZdcHitReconstructor::ZdcHitReconstructor(edm::ParameterSet const& conf):
  reco_(conf.getParameter<bool>("correctForTimeslew"),
	conf.getParameter<bool>("correctForPhaseContainment"),
	conf.getParameter<double>("correctionPhaseNS"),
	conf.getParameter<int>("recoMethod"),
	conf.getParameter<int>("lowGainOffset"),
	conf.getParameter<double>("lowGainFrac")),
  det_(DetId::Hcal),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
  correctTiming_(conf.getParameter<bool>("correctTiming")),
  setNoiseFlags_(conf.getParameter<bool>("setNoiseFlags")),
  setHSCPFlags_(conf.getParameter<bool>("setHSCPFlags")),
  setSaturationFlags_(conf.getParameter<bool>("setSaturationFlags")),
  setTimingTrustFlags_(conf.getParameter<bool>("setTimingTrustFlags")),
  dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")),
  AuxTSvec_(conf.getParameter<std::vector<int> >("AuxTSvec"))
  
{ 
  std::sort(AuxTSvec_.begin(),AuxTSvec_.end()); // sort vector in ascending TS order
  std::string subd=conf.getParameter<std::string>("Subdetector");
 
 if (setSaturationFlags_)
    {
      const edm::ParameterSet& pssat      = conf.getParameter<edm::ParameterSet>("saturationParameters");
      saturationFlagSetter_ = new HcalADCSaturationFlag(pssat.getParameter<int>("maxADCvalue"));
    }
  if (!strcasecmp(subd.c_str(),"ZDC")) {
    det_=DetId::Calo;
    subdet_=HcalZDCDetId::SubdetectorId;
    produces<ZDCRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"CALIB")) {
    subdet_=HcalOther;
    subdetOther_=HcalCalibration;
    produces<HcalCalibRecHitCollection>();
  } else {
    std::cout << "ZdcHitReconstructor is not associated with a specific subdetector!" << std::endl;
  }       
  
}

ZdcHitReconstructor::~ZdcHitReconstructor() {;
}
void ZdcHitReconstructor::beginRun(edm::Run&r, edm::EventSetup const & es){

   edm::ESHandle<HcalLongRecoParams> p;
   es.get<HcalLongRecoParamsRcd>().get(p);
   myobject = new HcalLongRecoParams(*p.product());
}

void ZdcHitReconstructor::endRun(edm::Run&r, edm::EventSetup const & es){
  if (myobject) delete myobject;
}
void ZdcHitReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  
  edm::ESHandle<HcalChannelQuality> p;
  eventSetup.get<HcalChannelQualityRcd>().get(p);
  HcalChannelQuality* myqual = new HcalChannelQuality(*p.product());

  edm::ESHandle<HcalSeverityLevelComputer> mycomputer;
  eventSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);
  const HcalSeverityLevelComputer* mySeverity = mycomputer.product();
  
  // define vectors to pass noiseTS and signalTS
  std::vector<unsigned int> mySignalTS;
  std::vector<unsigned int> myNoiseTS;
  
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
      DetId detcell=(DetId)cell;
      // check on cells to be ignored and dropped: (rof,20.Feb.09)
      const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
      if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;
	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	const HcalQIEShape* shape = conditions->getHcalShape (channelCoder);
	HcalCoderDb coder (*channelCoder, *shape);

// get db values for signalTSs and noiseTSs
   const HcalLongRecoParam* myParams = myobject->getValues(detcell);
   mySignalTS.clear();
   myNoiseTS.clear();
   mySignalTS = myParams->signalTS();
   myNoiseTS = myParams->noiseTS(); 

	rec->push_back(reco_.reconstruct(*i,myNoiseTS,mySignalTS,coder,calibrations));
	(rec->back()).setFlags(0);
	if (setSaturationFlags_)
	  saturationFlagSetter_->setSaturationFlag(rec->back(),*i);	

	// Set auxiliary flag with subset of digi information
	// ZDC aux flag can store non-contiguous set of values
        int auxflag=0;
	for (unsigned int xx=0; xx<AuxTSvec_.size() && xx<4;++xx)
	  {
	    if (AuxTSvec_[xx]<0 || AuxTSvec_[xx]>9) continue; // don't allow 
	    auxflag+=(i->sample(AuxTSvec_[xx]).adc())<<(7*xx); // store the time slices in the first 28 bits of aux, a set of 4 7-bit a dc values
	  }
	// bits 28 and 29 are reserved for capid of the first time slice saved in aux
	if (AuxTSvec_.size()>0)
	  auxflag+=((i->sample(AuxTSvec_[0]).capid())<<28);
	(rec->back()).setAux(auxflag);
     }
     // return result
     e.put(rec);     
   } // else if (det_==DetId::Calo...)

   delete myqual;
} // void HcalHitReconstructor::produce(...)
