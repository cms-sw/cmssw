// Module to import spy channel data into matching events

#include "DQM/SiStripMonitorHardware/interface/SiStripSpyEventMatcher.h"
#ifdef SiStripMonitorHardware_BuildEventMatchingCode

#include "boost/cstdint.hpp"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <memory>
#include <vector>

using edm::LogError;
using edm::LogInfo;

namespace sistrip {
  
  class SpyEventMatcherModule : public edm::EDFilter
  {
    public:
      SpyEventMatcherModule(const edm::ParameterSet& config);
      virtual ~SpyEventMatcherModule();
      virtual void beginJob();
      virtual bool filter(edm::Event& event, const edm::EventSetup& eventSetup);  
    private:
      void findL1IDandAPVAddress(const edm::Event& event, const SiStripFedCabling& cabling, uint32_t& l1ID, uint8_t& apvAddress) const;
      void copyData(const uint32_t eventId, const uint8_t apvAddress, const SpyEventMatcher::SpyEventList* matches, edm::Event& event,
                    const SiStripFedCabling& cabling) const;
      
      static const char* messageLabel_;
      const bool filterNonMatchingEvents_;
      const bool doMerge_;
      const edm::InputTag primaryStreamRawDataTag_;
      std::auto_ptr<SpyEventMatcher> spyEventMatcher_;
      std::auto_ptr<SpyUtilities> utils_;
  };
  
}

namespace sistrip {
  
  const char* SpyEventMatcherModule::messageLabel_ = "SiStripSpyDataMergeModule";
  
  SpyEventMatcherModule::SpyEventMatcherModule(const edm::ParameterSet& config)
    : filterNonMatchingEvents_( config.getParameter<bool>("FilterNonMatchingEvents") ),
      doMerge_( config.getParameter<bool>("MergeData") ),
      primaryStreamRawDataTag_( config.getParameter<edm::InputTag>("PrimaryEventRawDataTag") ),
      spyEventMatcher_(new SpyEventMatcher(config)),
      utils_(new SpyUtilities)
  {
    if (doMerge_) {
      produces<FEDRawDataCollection>("RawSpyData");
      produces< std::vector<uint32_t> >("SpyTotalEventCount");
      produces< std::vector<uint32_t> >("SpyL1ACount");
      produces< std::vector<uint32_t> >("SpyAPVAddress");
      produces< edm::DetSetVector<SiStripRawDigi> >("SpyScope");
      produces< edm::DetSetVector<SiStripRawDigi> >("SpyPayload");
      produces< edm::DetSetVector<SiStripRawDigi> >("SpyReordered");
      produces< edm::DetSetVector<SiStripRawDigi> >("SpyVirginRaw");
    }
  }
  
  SpyEventMatcherModule::~SpyEventMatcherModule()
  {
  }
  
  void SpyEventMatcherModule::beginJob()
  {
    spyEventMatcher_->initialize();
  }
  
  bool SpyEventMatcherModule::filter(edm::Event& event, const edm::EventSetup& eventSetup)
  {
    const SiStripFedCabling& cabling = *(utils_->getCabling(eventSetup));
    uint8_t apvAddress = 0;
    uint32_t eventId = 0;
    try {
      findL1IDandAPVAddress(event,cabling,eventId,apvAddress);
    } catch (const cms::Exception& e) {
      LogError(messageLabel_) << e.what();
      return ( filterNonMatchingEvents_ ? false : true );
    }
    const SpyEventMatcher::SpyEventList* matches = spyEventMatcher_->matchesForEvent(eventId,apvAddress);
    if (matches) {
      if (doMerge_) {
        copyData(eventId,apvAddress,matches,event,cabling);
      }
      return true;
    } else {
      return ( filterNonMatchingEvents_ ? false : true );
    }
  }
  
  void SpyEventMatcherModule::findL1IDandAPVAddress(const edm::Event& event, const SiStripFedCabling& cabling, uint32_t& l1ID, uint8_t& apvAddress) const
  {
    edm::Handle<FEDRawDataCollection> fedRawDataHandle;
    event.getByLabel(primaryStreamRawDataTag_,fedRawDataHandle);
    const FEDRawDataCollection& fedRawData = *fedRawDataHandle;
    for (std::vector<uint16_t>::const_iterator iFedId = cabling.feds().begin(); iFedId != cabling.feds().end(); ++iFedId) {
      const FEDRawData& data = fedRawData.FEDData(*iFedId);
      if ( (!data.data()) || (!data.size()) ) {
        LogDebug(messageLabel_) << "Failed to get FED data for FED ID " << *iFedId;
        continue;
      }
      std::auto_ptr<FEDBuffer> buffer;
      try {
        buffer.reset(new FEDBuffer(data.data(),data.size()));
      } catch (const cms::Exception& e) {
        LogDebug(messageLabel_) << "Failed to build FED buffer for FED ID " << *iFedId << ". Exception was " << e.what();
        continue;
      }
      if (!buffer->doChecks()) {
        LogDebug(messageLabel_) << "Buffer check failed for FED ID " << *iFedId;
        continue;
      }
      l1ID = buffer->daqLvl1ID();
      apvAddress = buffer->trackerSpecialHeader().apveAddress();
      if (apvAddress != 0) {
        return;
      } else {
        if (buffer->trackerSpecialHeader().headerType() != HEADER_TYPE_FULL_DEBUG) {
          continue;
        }
        const FEDFullDebugHeader* header = dynamic_cast<const FEDFullDebugHeader*>(buffer->feHeader());
        const std::vector<FedChannelConnection>& connections = cabling.connections(*iFedId);
        for (std::vector<FedChannelConnection>::const_iterator iConn = connections.begin(); iConn != connections.end(); ++iConn) {
          if (!iConn->isConnected()) {
            continue;
          }
          if ( !buffer->channelGood(iConn->fedCh()) ) {
            continue;
          } else {
            apvAddress = header->feUnitMajorityAddress(iConn->fedCh()/FEDCH_PER_FEUNIT);
            return;
          }
        }
      }
    }
    //if we haven't already found an acceptable alternative, throw an exception
    throw cms::Exception(messageLabel_) << "Failed to get L1ID/APV address from any FED";
  }
  
  void SpyEventMatcherModule::copyData(const uint32_t eventId, const uint8_t apvAddress, const SpyEventMatcher::SpyEventList* matches, edm::Event& event,
                                       const SiStripFedCabling& cabling) const
  {
    SpyEventMatcher::SpyDataCollections matchedCollections;
    spyEventMatcher_->getMatchedCollections(eventId,apvAddress,matches,cabling,matchedCollections);
    if (matchedCollections.rawData.get()) event.put(matchedCollections.rawData,"RawSpyData");
    if (matchedCollections.totalEventCounters.get()) event.put(matchedCollections.totalEventCounters,"SpyTotalEventCount");
    if (matchedCollections.l1aCounters.get()) event.put(matchedCollections.l1aCounters,"SpyL1ACount");
    if (matchedCollections.apvAddresses.get()) event.put(matchedCollections.apvAddresses,"SpyAPVAddress");
    if (matchedCollections.scopeDigis.get()) event.put(matchedCollections.scopeDigis,"SpyScope");
    if (matchedCollections.payloadDigis.get()) event.put(matchedCollections.payloadDigis,"SpyPayload");
    if (matchedCollections.reorderedDigis.get()) event.put(matchedCollections.reorderedDigis,"SpyReordered");
    if (matchedCollections.virginRawDigis.get()) event.put(matchedCollections.virginRawDigis,"SpyVirginRaw");
  }
  
}

#include "FWCore/Framework/interface/MakerMacros.h"
typedef sistrip::SpyEventMatcherModule SiStripSpyEventMatcherModule;
DEFINE_FWK_MODULE(SiStripSpyEventMatcherModule);

#endif //SiStripMonitorHardware_BuildEventMatchingCode
