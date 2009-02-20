using namespace std;
#include "HcalHitReconstructor.h"
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
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

#include <iostream>

/*  Hcal Hit reconstructor allows for CaloRecHits with status words */

HcalHitReconstructor::HcalHitReconstructor(edm::ParameterSet const& conf):
  reco_(conf.getParameter<int>("firstSample"),
	conf.getParameter<int>("samplesToAdd"),
	conf.getParameter<bool>("correctForTimeslew"),
	conf.getParameter<bool>("correctForPhaseContainment"),
	conf.getParameter<double>("correctionPhaseNS")),
  det_(DetId::Hcal),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
  channelStatusToDrop_(conf.getUntrackedParameter<std::vector<std::string> >("channelStatusesToDrop",std::vector<std::string>()))
{
  std::string subd=conf.getParameter<std::string>("Subdetector");
  hbheFlagSetter_=0;
  hfdigibit_=0;
  hfrechitbit_=0;

  if (!strcasecmp(subd.c_str(),"HBHE")) {
    subdet_=HcalBarrel;
    const edm::ParameterSet& psdigi  =conf.getParameter<edm::ParameterSet>("flagParameters");
    hbheFlagSetter_=new HBHEStatusBitSetter(psdigi.getParameter<double>("nominalPedestal"),
					    psdigi.getParameter<double>("hitEnergyMinimum"),
					    psdigi.getParameter<int>("hitMultiplicityThreshold"),
					    psdigi.getParameter<std::vector<edm::ParameterSet> >("pulseShapeParameterSets"));
    produces<HBHERecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"HO")) {
    subdet_=HcalOuter;
    produces<HORecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"HF")) {
    subdet_=HcalForward;
    // eventually move these outside if loop, when all cases contain such parameter sets?
    const edm::ParameterSet& psdigi  =conf.getParameter<edm::ParameterSet>("digistat");
    const edm::ParameterSet& psrechit=conf.getParameter<edm::ParameterSet>("rechitstat");
    hfdigibit_=new HcalHFStatusBitFromDigis(psdigi.getParameter<int>("HFpulsetimemin"),
					    psdigi.getParameter<int>("HFpulsetimemax"),
					    psdigi.getParameter<double>("HFratio_beforepeak"),
					    psdigi.getParameter<double>("HFratio_afterpeak"),
					    HcalCaloFlagLabels::HFDigiTime);
    hfrechitbit_=new HcalHFStatusBitFromRecHits(psrechit.getParameter<double>("HFlongshortratio"),
						HcalCaloFlagLabels::HFLongShort);
    produces<HFRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"ZDC")) {
    det_=DetId::Calo;
    subdet_=HcalZDCDetId::SubdetectorId;
    produces<ZDCRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"CALIB")) {
    subdet_=HcalOther;
    subdetOther_=HcalCalibration;
    produces<HcalCalibRecHitCollection>();
  } else {
    std::cout << "HcalHitReconstructor is not associated with a specific subdetector!" << std::endl;
  }       
  
}

HcalHitReconstructor::~HcalHitReconstructor() {
  if (hbheFlagSetter_)  delete hbheFlagSetter_;
  if (hfdigibit_)       delete hfdigibit_;
  if (hfrechitbit_)     delete hfrechitbit_;
}

void HcalHitReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  const HcalQIEShape* shape = conditions->getHcalShape (); // this one is generic
  
  edm::ESHandle<HcalChannelQuality> p;
  eventSetup.get<HcalChannelQualityRcd>().get(p);
  HcalChannelQuality* myqual = new HcalChannelQuality(*p.product());

  edm::ESHandle<HcalSeverityLevelComputer> mycomputer;
  eventSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);
  const HcalSeverityLevelComputer* mySeverity = mycomputer.product();

  
  if (det_==DetId::Hcal) {
    if (subdet_==HcalBarrel || subdet_==HcalEndcap) {
      edm::Handle<HBHEDigiCollection> digi;
      
      e.getByLabel(inputLabel_,digi);
      
      // create empty output
      std::auto_ptr<HBHERecHitCollection> rec(new HBHERecHitCollection);
      rec->reserve(digi->size());
      // run the algorithm
      hbheFlagSetter_->Clear();
      HBHEDigiCollection::const_iterator i;
      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;
	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);
	rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	(rec->back()).setFlags(0);
	hbheFlagSetter_->SetFlagsFromDigi(rec->back(),*i);
      }
      hbheFlagSetter_->SetFlagsFromRecHits(*rec);
      // return result
      e.put(rec);
    } else if (subdet_==HcalOuter) {
      edm::Handle<HODigiCollection> digi;
      e.getByLabel(inputLabel_,digi);
      
      // create empty output
      std::auto_ptr<HORecHitCollection> rec(new HORecHitCollection);
      rec->reserve(digi->size());
      // run the algorithm
      HODigiCollection::const_iterator i;
      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;
	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);
	rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	(rec->back()).setFlags(0);
      }
      // return result
      e.put(rec);    
    } else if (subdet_==HcalForward) {
      edm::Handle<HFDigiCollection> digi;
      e.getByLabel(inputLabel_,digi);
      ///////////////////////////////////////////////////////////////// HF
      // create empty output
      std::auto_ptr<HFRecHitCollection> rec(new HFRecHitCollection);
      rec->reserve(digi->size());
      // run the algorithm
      HFDigiCollection::const_iterator i;

      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;
	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);
	rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	(rec->back()).setFlags(0);
	// This calls the code for setting the low HF flag bit (bit 0)
	hfdigibit_->hfSetFlagFromDigi(rec->back(),*i);
      }
      // This sets HF flag bit 1
      hfrechitbit_->hfSetFlagFromRecHits(*rec);

      // return result
      e.put(rec);     
    } else if (subdet_==HcalOther && subdetOther_==HcalCalibration) {
      edm::Handle<HcalCalibDigiCollection> digi;
      e.getByLabel(inputLabel_,digi);
      
      // create empty output
      std::auto_ptr<HcalCalibRecHitCollection> rec(new HcalCalibRecHitCollection);
      rec->reserve(digi->size());
      // run the algorithm
      HcalCalibDigiCollection::const_iterator i;
      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalCalibDetId cell = i->id();
	DetId detcell=(DetId)cell;
	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);
	rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	//(rec->back()).setFlags(0); // Not yet implemented for HcalCalibRecHit
      }
      // return result
      e.put(rec);     
    }
  } else if (det_==DetId::Calo && subdet_==HcalZDCDetId::SubdetectorId) {
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

      const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
      const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
      HcalCoderDb coder (*channelCoder, *shape);
      rec->push_back(reco_.reconstruct(*i,coder,calibrations));
      (rec->back()).setFlags(0);
    }
    // return result
    e.put(rec);     
  } // else if (det_==DetId::Calo...)

  delete myqual;
} // void HcalHitReconstructor::produce(...)
