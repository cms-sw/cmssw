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
  correctTiming_(conf.getParameter<bool>("correctTiming")),
  setNoiseFlags_(conf.getParameter<bool>("setNoiseFlags")),
  setHSCPFlags_(conf.getParameter<bool>("setHSCPFlags")),
  setSaturationFlags_(conf.getParameter<bool>("setSaturationFlags")),
  setTimingTrustFlags_(conf.getParameter<bool>("setTimingTrustFlags")),
  dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed"))
  
{
  std::string subd=conf.getParameter<std::string>("Subdetector");
  hbheFlagSetter_=0;
  hbheHSCPFlagSetter_=0;
  hfdigibit_=0;
  hfrechitbit_=0;
  saturationFlagSetter_=0;
  HFTimingTrustFlagSetter_=0;

  if (setSaturationFlags_)
    {
      const edm::ParameterSet& pssat      = conf.getParameter<edm::ParameterSet>("saturationParameters");
      saturationFlagSetter_ = new HcalADCSaturationFlag(pssat.getParameter<int>("maxADCvalue"));
    }

  if (!strcasecmp(subd.c_str(),"HBHE")) {
    subdet_=HcalBarrel;
    if (setNoiseFlags_)
      {
	const edm::ParameterSet& psdigi    =conf.getParameter<edm::ParameterSet>("flagParameters");
	hbheFlagSetter_=new HBHEStatusBitSetter(psdigi.getParameter<double>("nominalPedestal"),
						psdigi.getParameter<double>("hitEnergyMinimum"),
						psdigi.getParameter<int>("hitMultiplicityThreshold"),
						psdigi.getParameter<std::vector<edm::ParameterSet> >("pulseShapeParameterSets"));
      } // if (setNoiseFlags_)
    if (setHSCPFlags_)
      {
	const edm::ParameterSet& psHSCP = conf.getParameter<edm::ParameterSet>("hscpParameters");
	hbheHSCPFlagSetter_ = new HBHETimeProfileStatusBitSetter(psHSCP.getParameter<double>("r1Min"),
								 psHSCP.getParameter<double>("r1Max"),
								 psHSCP.getParameter<double>("r2Min"),
								 psHSCP.getParameter<double>("r2Max"),
								 psHSCP.getParameter<double>("fracLeaderMin"),
								 psHSCP.getParameter<double>("fracLeaderMax"),
								 psHSCP.getParameter<double>("slopeMin"),
								 psHSCP.getParameter<double>("slopeMax"),
								 psHSCP.getParameter<double>("outerMin"),
								 psHSCP.getParameter<double>("outerMax"),
								 psHSCP.getParameter<double>("TimingEnergyThreshold"));
      } // if (setHSCPFlags_) 
    produces<HBHERecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"HO")) {
    subdet_=HcalOuter;
    produces<HORecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"HF")) {
    subdet_=HcalForward;

    if (setTimingTrustFlags_) {
      
      const edm::ParameterSet& pstrust      = conf.getParameter<edm::ParameterSet>("hfTimingTrustParameters");
      HFTimingTrustFlagSetter_=new HFTimingTrustFlag(pstrust.getParameter<int>("hfTimingTrustLevel1"),
						     pstrust.getParameter<int>("hfTimingTrustLevel2"));
    }


    if (setNoiseFlags_)
      {
	const edm::ParameterSet& psdigi    =conf.getParameter<edm::ParameterSet>("digistat");
	const edm::ParameterSet& psrechit  =conf.getParameter<edm::ParameterSet>("rechitstat");
	hfdigibit_=new HcalHFStatusBitFromDigis(psdigi.getParameter<int>("HFpulsetimemin"),
						psdigi.getParameter<int>("HFpulsetimemax"),
						psdigi.getParameter<double>("HFratio_beforepeak"),
						psdigi.getParameter<double>("HFratio_afterpeak"),
						psdigi.getParameter<int>("HFadcthreshold"));
	hfrechitbit_=new HcalHFStatusBitFromRecHits(psrechit.getParameter<double>("HFlongshortratio"),
						    psrechit.getParameter<double>("HFthresholdET"));
      }
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
  if (hbheFlagSetter_)        delete hbheFlagSetter_;
  if (hfdigibit_)             delete hfdigibit_;
  if (hfrechitbit_)           delete hfrechitbit_;
  if (hbheHSCPFlagSetter_)    delete hbheHSCPFlagSetter_;
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
      if (setNoiseFlags_) hbheFlagSetter_->Clear();
      HBHEDigiCollection::const_iterator i;
      std::vector<HBHEDataFrame> HBDigis;
      std::vector<int> RecHitIndex;

      // Vote on majority TS0 CapId
      int favorite_capid = 0; 
      if (correctTiming_) {
        long capid_votes[4] = {0,0,0,0};
        for (i=digi->begin(); i!=digi->end(); i++) {
          capid_votes[(*i)[0].capid()]++;
        }
        for (int k = 0; k < 4; k++)
          if (capid_votes[k] > capid_votes[favorite_capid])
            favorite_capid = k;
      }

      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;

	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);
	rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	(rec->back()).setFlags(0);
	if (setNoiseFlags_)
	  hbheFlagSetter_->SetFlagsFromDigi(rec->back(),*i);
	if (setSaturationFlags_)
	  saturationFlagSetter_->setSaturationFlag(rec->back(),*i);
	if (correctTiming_)
	  HcalTimingCorrector::Correct(rec->back(), *i, favorite_capid);
	if (setHSCPFlags_ && i->id().ietaAbs()<16)
	  {
	    double DigiEnergy=0;
            for(int j=0; j!=i->size(); DigiEnergy += i->sample(j++).nominal_fC());
            if(DigiEnergy > hbheHSCPFlagSetter_->EnergyThreshold())
              {
                HBDigis.push_back(*i);
                RecHitIndex.push_back(rec->size()-1);
              }
	    
	  } // if (set HSCPFlags_ && |ieta|<16)
      } // loop over HBHE digis


      if (setNoiseFlags_) hbheFlagSetter_->SetFlagsFromRecHits(*rec);
      if (setHSCPFlags_)  hbheHSCPFlagSetter_->hbheSetTimeFlagsFromDigi(rec.get(), HBDigis, RecHitIndex);
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

      // Vote on majority TS0 CapId
      int favorite_capid = 0; 
      if (correctTiming_) {
        long capid_votes[4] = {0,0,0,0};
        for (i=digi->begin(); i!=digi->end(); i++) {
          capid_votes[(*i)[0].capid()]++;
        }
        for (int k = 0; k < 4; k++)
          if (capid_votes[k] > capid_votes[favorite_capid])
            favorite_capid = k;
      }

      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;
	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);
	rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	(rec->back()).setFlags(0);
	if (setSaturationFlags_)
	  saturationFlagSetter_->setSaturationFlag(rec->back(),*i);
	if (correctTiming_)
	  HcalTimingCorrector::Correct(rec->back(), *i, favorite_capid);
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

      // Vote on majority TS0 CapId
      int favorite_capid = 0; 
      if (correctTiming_) {
        long capid_votes[4] = {0,0,0,0};
        for (i=digi->begin(); i!=digi->end(); i++) {
          capid_votes[(*i)[0].capid()]++;
        }
        for (int k = 0; k < 4; k++)
          if (capid_votes[k] > capid_votes[favorite_capid])
            favorite_capid = k;
      }

      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;
	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);
	rec->push_back(reco_.reconstruct(*i,coder,calibrations));
	(rec->back()).setFlags(0);
	// This calls the code for setting the HF noise bit determined from digi shape
	if (setNoiseFlags_) hfdigibit_->hfSetFlagFromDigi(rec->back(),*i);
	if (setSaturationFlags_)
	  saturationFlagSetter_->setSaturationFlag(rec->back(),*i);
	if (setTimingTrustFlags_)
	  HFTimingTrustFlagSetter_->setHFTimingTrustFlag(rec->back(),*i);
	if (correctTiming_)
	  HcalTimingCorrector::Correct(rec->back(), *i, favorite_capid);
      }
      // This sets HF noise bit determined from L/S rechit energy comparison
      if (setNoiseFlags_) hfrechitbit_->hfSetFlagFromRecHits(*rec);
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
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

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
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;
	    
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
