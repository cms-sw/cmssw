using namespace std;

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBObjectUnpacker.h"
#include "FWCore/EDProduct/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include <iostream>

using namespace raw;

namespace hcaltb
{

  HcalTBObjectUnpacker::HcalTBObjectUnpacker(edm::ParameterSet const& conf):
    triggerFed_(conf.getParameter<int>("HcalTriggerFED")),
    sdFed_(conf.getParameter<int>("HcalSlowDataFED")),
    tdcFed_(conf.getParameter<int>("HcalTDCFED"))
  {
    if (triggerFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack FED ";
      std::cout << triggerFed_ << endl;
    }

    if (sdFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack SlowData FED ";
      std::cout << sdFed_ << endl;
    }

    if (tdcFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack TDC FED ";
      std::cout << tdcFed_ << endl;
    }

    produces<HcalTBTriggerData>();
    produces<HcalTBRunData>();
    produces<HcalTBEventPosition>();
    produces<HcalTBTiming>();
  }

  // Virtual destructor needed.
  HcalTBObjectUnpacker::~HcalTBObjectUnpacker() { }  

  // Functions that gets called by framework every event
  void HcalTBObjectUnpacker::produce(edm::Event& e, const edm::EventSetup&)
  {
    // Step A: Get Inputs 
    edm::Handle<FEDRawDataCollection> rawraw;  
    edm::ProcessNameSelector s("PROD"); // HACK!
    e.get(s, rawraw);           

    // Step B: Create empty output
    std::auto_ptr<hcaltb::HcalTBTriggerData>
      trigd(new hcaltb::HcalTBTriggerData);

    std::auto_ptr<hcaltb::HcalTBRunData>
      rund(new hcaltb::HcalTBRunData);

    std::auto_ptr<hcaltb::HcalTBEventPosition>
      epd(new hcaltb::HcalTBEventPosition);

    std::auto_ptr<hcaltb::HcalTBTiming>
      tmgd(new hcaltb::HcalTBTiming);

    if (triggerFed_ >=0) {

      // Step C: unpack all requested FEDs
      const raw::FEDRawData& fed = rawraw->FEDData(triggerFed_);
      tdUnpacker_.unpack(fed,*trigd);
    }

    if (sdFed_ >=0) {

      // Step C: unpack all requested FEDs
      const raw::FEDRawData& fed = rawraw->FEDData(sdFed_);
      sdUnpacker_.unpack(fed, *rund, *epd);
    }

    if (tdcFed_ >=0) {

      // Step C: unpack all requested FEDs
      const raw::FEDRawData& fed = rawraw->FEDData(tdcFed_);
      tdcUnpacker_.unpack(fed, *epd, *tmgd);
    }

    // Step D: Put outputs into event
    e.put(trigd);
    e.put(rund);
    e.put(epd);
    e.put(tmgd);
  }

}
