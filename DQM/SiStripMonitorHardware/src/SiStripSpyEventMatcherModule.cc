// Module to import spy channel data into matching events

#include "DQM/SiStripMonitorHardware/interface/SiStripSpyEventMatcher.h"
#ifdef SiStripMonitorHardware_BuildEventMatchingCode

#include "FWCore/Utilities/interface/EDGetToken.h"
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

  class SpyEventMatcherModule : public edm::EDFilter {
  public:
    SpyEventMatcherModule(const edm::ParameterSet& config);
    ~SpyEventMatcherModule() override;
    void beginJob() override;
    bool filter(edm::Event& event, const edm::EventSetup& eventSetup) override;

  private:
    void findL1IDandAPVAddress(const edm::Event& event,
                               const SiStripFedCabling& cabling,
                               uint32_t& l1ID,
                               uint8_t& apvAddress) const;
    void copyData(const uint32_t eventId,
                  const uint8_t apvAddress,
                  const SpyEventMatcher::SpyEventList* matches,
                  edm::Event& event,
                  const SiStripFedCabling& cabling) const;

    static const char* messageLabel_;
    const bool filterNonMatchingEvents_;
    const bool doMerge_;
    const edm::InputTag primaryStreamRawDataTag_;
    edm::EDGetTokenT<FEDRawDataCollection> primaryStreamRawDataToken_;
    std::unique_ptr<SpyEventMatcher> spyEventMatcher_;
    std::unique_ptr<SpyUtilities> utils_;
  };

}  // namespace sistrip

namespace sistrip {

  const char* SpyEventMatcherModule::messageLabel_ = "SiStripSpyDataMergeModule";

  SpyEventMatcherModule::SpyEventMatcherModule(const edm::ParameterSet& config)
      : filterNonMatchingEvents_(config.getParameter<bool>("FilterNonMatchingEvents")),
        doMerge_(config.getParameter<bool>("MergeData")),
        primaryStreamRawDataTag_(config.getParameter<edm::InputTag>("PrimaryEventRawDataTag")),
        spyEventMatcher_(new SpyEventMatcher(config)),
        utils_(new SpyUtilities) {
    primaryStreamRawDataToken_ = consumes<FEDRawDataCollection>(primaryStreamRawDataTag_);
    if (doMerge_) {
      produces<FEDRawDataCollection>("RawSpyData");
      produces<std::vector<uint32_t> >("SpyTotalEventCount");
      produces<std::vector<uint32_t> >("SpyL1ACount");
      produces<std::vector<uint32_t> >("SpyAPVAddress");
      produces<edm::DetSetVector<SiStripRawDigi> >("SpyScope");
      produces<edm::DetSetVector<SiStripRawDigi> >("SpyPayload");
      produces<edm::DetSetVector<SiStripRawDigi> >("SpyReordered");
      produces<edm::DetSetVector<SiStripRawDigi> >("SpyVirginRaw");
    }
  }

  SpyEventMatcherModule::~SpyEventMatcherModule() {}

  void SpyEventMatcherModule::beginJob() { spyEventMatcher_->initialize(); }

  bool SpyEventMatcherModule::filter(edm::Event& event, const edm::EventSetup& eventSetup) {
    const SiStripFedCabling& cabling = *(utils_->getCabling(eventSetup));
    uint8_t apvAddress = 0;
    uint32_t eventId = 0;
    try {
      findL1IDandAPVAddress(event, cabling, eventId, apvAddress);
    } catch (const cms::Exception& e) {
      LogError(messageLabel_) << e.what();
      return (filterNonMatchingEvents_ ? false : true);
    }
    const SpyEventMatcher::SpyEventList* matches = spyEventMatcher_->matchesForEvent(eventId, apvAddress);
    if (matches) {
      if (doMerge_) {
        copyData(eventId, apvAddress, matches, event, cabling);
      }
      return true;
    } else {
      return (filterNonMatchingEvents_ ? false : true);
    }
  }

  void SpyEventMatcherModule::findL1IDandAPVAddress(const edm::Event& event,
                                                    const SiStripFedCabling& cabling,
                                                    uint32_t& l1ID,
                                                    uint8_t& apvAddress) const {
    edm::Handle<FEDRawDataCollection> fedRawDataHandle;
    event.getByToken(primaryStreamRawDataToken_, fedRawDataHandle);
    const FEDRawDataCollection& fedRawData = *fedRawDataHandle;
    for (auto iFedId = cabling.fedIds().begin(); iFedId != cabling.fedIds().end(); ++iFedId) {
      const FEDRawData& data = fedRawData.FEDData(*iFedId);
      const auto st_buffer = preconstructCheckFEDBuffer(data);
      if (FEDBufferStatusCode::SUCCESS != st_buffer) {
        LogInfo(messageLabel_) << "Failed to build FED buffer for FED ID " << *iFedId
                               << ". Exception was: An exception of category 'FEDBuffer' occurred.\n"
                               << st_buffer << " (see debug output for details)";
        continue;
      }
      FEDBuffer buffer{data};
      const auto st_chan = buffer.findChannels();
      if (FEDBufferStatusCode::SUCCESS != st_chan) {
        LogDebug(messageLabel_) << "Failed to build FED buffer for FED ID " << *iFedId << ". Exception was " << st_chan
                                << " (see above for more details)";
        continue;
      }
      if (!buffer.doChecks(true)) {
        LogDebug(messageLabel_) << "Buffer check failed for FED ID " << *iFedId;
        continue;
      }
      l1ID = buffer.daqLvl1ID();
      apvAddress = buffer.trackerSpecialHeader().apveAddress();
      if (apvAddress != 0) {
        return;
      } else {
        if (buffer.trackerSpecialHeader().headerType() != HEADER_TYPE_FULL_DEBUG) {
          continue;
        }
        const FEDFullDebugHeader* header = dynamic_cast<const FEDFullDebugHeader*>(buffer.feHeader());
        auto connections = cabling.fedConnections(*iFedId);
        for (auto iConn = connections.begin(); iConn != connections.end(); ++iConn) {
          if (!iConn->isConnected()) {
            continue;
          }
          if (!buffer.channelGood(iConn->fedCh(), true)) {
            continue;
          } else {
            apvAddress = header->feUnitMajorityAddress(iConn->fedCh() / FEDCH_PER_FEUNIT);
            return;
          }
        }
      }
    }
    //if we haven't already found an acceptable alternative, throw an exception
    throw cms::Exception(messageLabel_) << "Failed to get L1ID/APV address from any FED";
  }

  void SpyEventMatcherModule::copyData(const uint32_t eventId,
                                       const uint8_t apvAddress,
                                       const SpyEventMatcher::SpyEventList* matches,
                                       edm::Event& event,
                                       const SiStripFedCabling& cabling) const {
    SpyEventMatcher::SpyDataCollections matchedCollections;
    spyEventMatcher_->getMatchedCollections(eventId, apvAddress, matches, cabling, matchedCollections);
    if (matchedCollections.rawData.get())
      event.put(std::move(matchedCollections.rawData), "RawSpyData");
    if (matchedCollections.totalEventCounters.get())
      event.put(std::move(matchedCollections.totalEventCounters), "SpyTotalEventCount");
    if (matchedCollections.l1aCounters.get())
      event.put(std::move(matchedCollections.l1aCounters), "SpyL1ACount");
    if (matchedCollections.apvAddresses.get())
      event.put(std::move(matchedCollections.apvAddresses), "SpyAPVAddress");
    if (matchedCollections.scopeDigis.get())
      event.put(std::move(matchedCollections.scopeDigis), "SpyScope");
    if (matchedCollections.payloadDigis.get())
      event.put(std::move(matchedCollections.payloadDigis), "SpyPayload");
    if (matchedCollections.reorderedDigis.get())
      event.put(std::move(matchedCollections.reorderedDigis), "SpyReordered");
    if (matchedCollections.virginRawDigis.get())
      event.put(std::move(matchedCollections.virginRawDigis), "SpyVirginRaw");
  }

}  // namespace sistrip

#include "FWCore/Framework/interface/MakerMacros.h"
#include <cstdint>
typedef sistrip::SpyEventMatcherModule SiStripSpyEventMatcherModule;
DEFINE_FWK_MODULE(SiStripSpyEventMatcherModule);

#endif  //SiStripMonitorHardware_BuildEventMatchingCode
