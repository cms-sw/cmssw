//Class to produce a dummy SiStripEventSummary object so that spy channel data can be used with commissioning software. 
//Run types which need additional parameters from the trigger FED buffer or DAQ registers are not supported. 
//If an unsupported run type is used, an error message will be printed and parameters will be set to zero. 
//Author: Nick Cripps
//Date: 10/05/2010

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForRunType.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include <memory>
#include <string>
#include "boost/scoped_array.hpp"
#include "boost/cstdint.hpp"

using edm::LogError;
using edm::LogWarning;
using edm::LogInfo;

namespace sistrip {
  
  class SpyEventSummaryProducer : public edm::EDProducer
  {
    public:
      SpyEventSummaryProducer(const edm::ParameterSet& config);
      virtual ~SpyEventSummaryProducer();
      virtual void produce(edm::Event& event, const edm::EventSetup&) override;
    private:
      void warnAboutUnsupportedRunType();
      static const char* messageLabel_;
      const edm::InputTag rawDataTag_;
      const sistrip::RunType runType_;
  };
  
}

namespace sistrip {
  
  const char* SpyEventSummaryProducer::messageLabel_ = "SiStripSpyEventSummaryProducer";
  
  SpyEventSummaryProducer::SpyEventSummaryProducer(const edm::ParameterSet& config)
    : rawDataTag_(config.getParameter<edm::InputTag>("RawDataTag")),
      runType_(sistrip::RunType(config.getParameter<uint32_t>("RunType")))
  {
    produces<SiStripEventSummary>();
    warnAboutUnsupportedRunType();
  }
  
  SpyEventSummaryProducer::~SpyEventSummaryProducer() {}
  
  void SpyEventSummaryProducer::produce(edm::Event& event, const edm::EventSetup&)
  {
    warnAboutUnsupportedRunType();
    
    //get the event number and Bx counter from the first valud FED buffer
    edm::Handle<FEDRawDataCollection> rawDataHandle;
    event.getByLabel(rawDataTag_,rawDataHandle);
    const FEDRawDataCollection& rawData = *rawDataHandle;
    bool fedFound = false;
    uint32_t fedEventNumber = 0;
    uint32_t fedBxNumber = 0;
    for (uint16_t fedId = sistrip::FED_ID_MIN; fedId <= sistrip::FED_ID_MAX; ++fedId) {
      const FEDRawData& fedData = rawData.FEDData(fedId);
      if (fedData.size() && fedData.data()) {
        std::auto_ptr<sistrip::FEDBufferBase> pBuffer;
        try {
          pBuffer.reset(new sistrip::FEDBufferBase(fedData.data(),fedData.size()));
        } catch (const cms::Exception& e) {
          LogInfo(messageLabel_) << "Skipping FED " << fedId << " because of exception: " << e.what();
          continue;
        }
        fedEventNumber = pBuffer->daqLvl1ID();
        fedBxNumber = pBuffer->daqBXID();
        fedFound = true;
        break;
      }
    }
    if (!fedFound) {
      LogError(messageLabel_) << "No SiStrip FED data found in raw data.";
      return;
    }
    
    //create summary object
    std::auto_ptr<SiStripEventSummary> pSummary(new SiStripEventSummary);
    //set the trigger FED number to zero to indicate that it doesn't exist
    pSummary->triggerFed(0);
    //set the event number and Bx from the FED packets
    pSummary->event(fedEventNumber);
    pSummary->bx(fedBxNumber);
    //create a fake trigger FED buffer to take comissioning parameters from
    const int maxTriggerFedBufferSize = 84;
    boost::scoped_array<uint32_t> fakeTriggerFedData(new uint32_t[maxTriggerFedBufferSize]);
    for (uint8_t i=0; i<maxTriggerFedBufferSize; ++i) {
      fakeTriggerFedData[i] = 0;
    }
    //set the FED readout mode to virgin raw
    fakeTriggerFedData[15] = 1;
    //set the spill number
    fakeTriggerFedData[0] = 0;
    //set the number of data senders
    fakeTriggerFedData[20] = 1;
    //set the run type
    fakeTriggerFedData[10] = runType_;
    //fill the summarry using trigger FED buffer  with no data
    pSummary->commissioningInfo(fakeTriggerFedData.get(),fedEventNumber);
    
    //store in event
    event.put(pSummary);
  }
  
  void SpyEventSummaryProducer::warnAboutUnsupportedRunType()
  {
    switch(runType_) {
      case sistrip::DAQ_SCOPE_MODE:
      case sistrip::PHYSICS:
      case sistrip::PHYSICS_ZS:
      case sistrip::PEDESTALS:
      case sistrip::MULTI_MODE:
      case sistrip::PEDS_ONLY:
      case sistrip::NOISE:
      case sistrip::PEDS_FULL_NOISE:
      case sistrip::UNKNOWN_RUN_TYPE:
      case sistrip::UNDEFINED_RUN_TYPE:
        break;
      case sistrip::CALIBRATION:
      case sistrip::CALIBRATION_DECO:
      case sistrip::CALIBRATION_SCAN:
      case sistrip::CALIBRATION_SCAN_DECO:
      case sistrip::APV_LATENCY:
      case sistrip::OPTO_SCAN:
      case sistrip::APV_TIMING:
      case sistrip::FED_TIMING:
      case sistrip::FINE_DELAY:
      case sistrip::FINE_DELAY_PLL:
      case sistrip::FINE_DELAY_TTC:
      case sistrip::FAST_CABLING:
      case sistrip::FED_CABLING:
      case sistrip::QUITE_FAST_CABLING:
      case sistrip::VPSP_SCAN:
        LogWarning(messageLabel_) << "Unsupported run type: " << runType_ << ". Parameters need to be set from real trigger FED. Parameters will be set to 0.";
        break;
    }
  }
  
}

typedef sistrip::SpyEventSummaryProducer SiStripSpyEventSummaryProducer;
DEFINE_FWK_MODULE(SiStripSpyEventSummaryProducer);
