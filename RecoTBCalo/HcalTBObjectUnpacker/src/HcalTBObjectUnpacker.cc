using namespace std;

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBObjectUnpacker.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include <iostream>


  HcalTBObjectUnpacker::HcalTBObjectUnpacker(edm::ParameterSet const& conf):
    triggerFed_(conf.getUntrackedParameter<int>("HcalTriggerFED",-1)),
    sdFed_(conf.getUntrackedParameter<int>("HcalSlowDataFED",-1)),
    spdFed_(conf.getUntrackedParameter<int>("HcalSourcePositionFED",-1)),
    tdcFed_(conf.getUntrackedParameter<int>("HcalTDCFED",-1)),
    tdcUnpacker_(conf.getUntrackedParameter<bool>("IncludeUnmatchedHits",false)),
    doRunData_(false),doTriggerData_(false),doEventPosition_(false),doTiming_(false),doSourcePos_(false)
  {
    if (triggerFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack FED ";
      std::cout << triggerFed_ << endl;
      doTriggerData_=true;
    }

    if (sdFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack SlowData FED ";
      std::cout << sdFed_ << endl;
      doRunData_=true;
      doEventPosition_=true; // at least the table
    }

    if (tdcFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack TDC FED ";
      std::cout << tdcFed_ << endl;
      doTiming_=true;
      doEventPosition_=true; // at least the WC
    }

    if (spdFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack Source Position Data FED ";
      std::cout << spdFed_ << endl;
      doSourcePos_=true;
    }

    if (doTriggerData_) produces<HcalTBTriggerData>();
    if (doRunData_) produces<HcalTBRunData>();
    if (doEventPosition_) produces<HcalTBEventPosition>();
    if (doTiming_) produces<HcalTBTiming>();
    if (doSourcePos_) produces<HcalSourcePositionData>();
  }

  // Virtual destructor needed.
  HcalTBObjectUnpacker::~HcalTBObjectUnpacker() { }  

  // Functions that gets called by framework every event
  void HcalTBObjectUnpacker::produce(edm::Event& e, const edm::EventSetup&)
  {
    // Step A: Get Inputs 
    edm::Handle<FEDRawDataCollection> rawraw;  
    //    edm::ProcessNameSelector s("PROD"); // HACK!
    e.getByType(rawraw);           

    // Step B: Create empty output    
    std::auto_ptr<HcalTBTriggerData>
      trigd(new HcalTBTriggerData);

    std::auto_ptr<HcalTBRunData>
      rund(new HcalTBRunData);

    std::auto_ptr<HcalTBEventPosition>
      epd(new HcalTBEventPosition);

    std::auto_ptr<HcalTBTiming>
      tmgd(new HcalTBTiming);

    std::auto_ptr<HcalSourcePositionData>
      spd(new HcalSourcePositionData);
    
    if (triggerFed_ >=0) {
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(triggerFed_);
      tdUnpacker_.unpack(fed,*trigd);
    }

    if (sdFed_ >=0) {
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(sdFed_);
      sdUnpacker_.unpack(fed, *rund, *epd);
    }

    if (tdcFed_ >=0) {
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(tdcFed_);
      tdcUnpacker_.unpack(fed, *epd, *tmgd);
    }

    if (spdFed_ >=0) {      
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(spdFed_);
      spdUnpacker_.unpack(fed, *spd);
    }

    // Step D: Put outputs into event
    if (doTriggerData_) e.put(trigd);
    if (doRunData_) e.put(rund);
    if (doEventPosition_) e.put(epd);
    if (doTiming_) e.put(tmgd);
    if (doSourcePos_) e.put(spd);
  }
