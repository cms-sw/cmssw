#include "DQM/SiStripMonitorHardware/interface/SiStripSpyEventMatcher.h"
#ifdef SiStripMonitorHardware_BuildEventMatchingCode

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>

using edm::LogInfo;
using edm::LogWarning;
using edm::LogError;

namespace sistrip {
  
  const char* SpyEventMatcher::mlLabel_ = "SpyEventMatcher";
  
  SpyEventMatcher::EventKey::EventKey(const uint32_t eventId, const uint8_t apvAddress)
    : eventId_(eventId), apvAddress_(apvAddress) {}
  
  SpyEventMatcher::~SpyEventMatcher() {}
  
  SpyEventMatcher::SpyEventMatcher(const edm::ParameterSet& config)
    : rawDataTag_(config.getParameter<edm::InputTag>("RawSpyDataTag")),
      totalEventCountersTag_(config.getParameter<edm::InputTag>("SpyTotalEventCountersTag")),
      l1aCountersTag_(config.getParameter<edm::InputTag>("SpyL1ACountersTag")),
      apvAddressesTag_(config.getParameter<edm::InputTag>("SpyAPVAddressesTag")),
      scopeDigisTag_(config.getParameter<edm::InputTag>("SpyScopeDigisTag")),
      payloadDigisTag_(config.getParameter<edm::InputTag>("SpyPayloadDigisTag")),
      reorderedDigisTag_(config.getParameter<edm::InputTag>("SpyReorderedDigisTag")),
      virginRawDigisTag_(config.getParameter<edm::InputTag>("SpyVirginRawDigisTag")),
      source_(constructSource(config.getParameter<edm::ParameterSet>("SpySource")))
  {}
  
  std::auto_ptr<SpyEventMatcher::Source> SpyEventMatcher::constructSource(const edm::ParameterSet& sourceConfig)
  {
    const edm::VectorInputSourceFactory* sourceFactory = edm::VectorInputSourceFactory::get();
    edm::InputSourceDescription description;
    return sourceFactory->makeVectorInputSource(sourceConfig, description);
  }
  
  void SpyEventMatcher::initialize()
  {
    //add spy events to the map until there are none left
    while ( addNextEventToMap() ) {;}
    //debug
    std::ostringstream ss;
    ss << "Events with possible matches (eventID,apvAddress): ";
    for (std::map<EventKey,SpyEventList>::const_iterator iSpyEvent = eventMatches_.begin(); iSpyEvent != eventMatches_.end(); ++iSpyEvent) {
      ss << "(" << iSpyEvent->first.eventId() << "," << uint16_t(iSpyEvent->first.apvAddress()) << ") ";
    }
    LogDebug(mlLabel_) << ss.str();
  }
  
  bool SpyEventMatcher::addNextEventToMap()
  {
    SpyEventPtr nextSpyEvent = readNextEvent();
    if (!nextSpyEvent) {
      LogInfo(mlLabel_) << "no more spy events";
      return false;
    }
    edm::EventID spyEventId = nextSpyEvent->id();
    const std::vector<uint32_t>& totalEventCounters = *getProduct< std::vector<uint32_t> >(nextSpyEvent,totalEventCountersTag_);
    const std::vector<uint32_t>& l1aCounters = *getProduct< std::vector<uint32_t> >(nextSpyEvent,l1aCountersTag_);
    const std::vector<uint32_t>& apvAddresses = *getProduct< std::vector<uint32_t> >(nextSpyEvent,apvAddressesTag_);
    //loop over all FEDs. Maps should have same content and be in order so, avoid searches by iterating (and checking keys match)
    //add all possible event keys to the map
    std::vector<uint32_t>::const_iterator iTotalEventCount = totalEventCounters.begin();
    std::vector<uint32_t>::const_iterator iL1ACount = l1aCounters.begin();
    std::vector<uint32_t>::const_iterator iAPVAddress = apvAddresses.begin();
    //for debug
    std::map<EventKey,uint16_t> fedCounts;
    for (;
         ( (iTotalEventCount != totalEventCounters.end()) && (iL1ACount != l1aCounters.end()) && (iAPVAddress != apvAddresses.end()) ); 
         (++iTotalEventCount, ++iL1ACount, ++iAPVAddress)
        ){
      for (uint32_t eventId = *iTotalEventCount+1; eventId <= *iL1ACount+1; ++eventId) {
        EventKey key(eventId,*iAPVAddress);
        eventMatches_[key].insert(spyEventId);
        fedCounts[key]++;
      }
    }
    //for debug
    std::ostringstream ss;
    ss << "Spy event " << spyEventId.event() << " matches (eventID,apvAddress,nFEDs): ";
    for (std::map<EventKey,uint16_t>::const_iterator iEventFEDCount = fedCounts.begin(); iEventFEDCount != fedCounts.end(); ++iEventFEDCount) {
      ss << "(" << iEventFEDCount->first.eventId() << "," << uint16_t(iEventFEDCount->first.apvAddress()) << "," << iEventFEDCount->second << ") ";
    }
    LogDebug(mlLabel_) << ss.str();
    return true;
  }
  
  SpyEventMatcher::SpyEventPtr SpyEventMatcher::readNextEvent()
  {
    Source::EventPrincipalVector events;
    unsigned int file;
    source_->readManySequential(1,events,file);
    if (events.size()) {
      if (events.size() != 1) {
        LogError(mlLabel_) << events.size() << " events read. Ignoring the rest";
      }
      return events[0];
    } else {
      return SpyEventPtr();
    }
  }
  
  SpyEventMatcher::SpyEventPtr SpyEventMatcher::readSpecificEvent(const edm::EventID& id)
  {
    Source::EventPrincipalVector events;
    std::vector<edm::EventID> ids(1,id);
    source_->readManySpecified(ids,events);
    if (events.size()) {
      if (events.size() != 1) {
        LogError(mlLabel_) << events.size() << " events read for ID " << id.event() << ". Ignoring the rest";
      }
      return events[0];
    } else {
      throw cms::Exception(mlLabel_) << "Spy event " << id.event() << " is in the map but was not found";
    }
  }
  
  const SpyEventMatcher::SpyEventList* SpyEventMatcher::matchesForEvent(const uint32_t eventId, const uint8_t apvAddress) const
  {
    EventKey eventKey(eventId,apvAddress);
    std::map<EventKey,SpyEventList>::const_iterator iMatch = eventMatches_.find(eventKey);
    if (iMatch == eventMatches_.end()) {
      LogDebug(mlLabel_) << "No match found for event " << eventId << " with APV address " << uint16_t(apvAddress);
      return NULL;
    }
    else {
      std::ostringstream ss;
      ss << "Found matches to event " << eventId << " with address " << uint16_t(apvAddress) << " in spy events ";
      for (SpyEventList::const_iterator iMatchingSpyEvent = iMatch->second.begin(); iMatchingSpyEvent != iMatch->second.end(); ++iMatchingSpyEvent) {
        ss << iMatchingSpyEvent->event() << " ";
      }
      LogInfo(mlLabel_) << ss.str();
      return &(iMatch->second);
    }
  }
  
  void SpyEventMatcher::getMatchedCollections(const uint32_t eventId, const uint8_t apvAddress,
                                            const SpyEventList* matchingEvents, const SiStripFedCabling& cabling,
                                            SpyDataCollections& collectionsToCreate)
  {
    if (!matchingEvents) return;
    FEDRawDataCollection outputRawData;
    std::vector<uint32_t> outputTotalEventCounters(sistrip::FED_ID_MAX+1);
    std::vector<uint32_t> outputL1ACounters(sistrip::FED_ID_MAX+1);
    std::vector<uint32_t> outputAPVAddresses(sistrip::FED_ID_MAX+1);
    std::vector< edm::DetSet<SiStripRawDigi> >* outputScopeDigisVector = NULL;
    std::vector< edm::DetSet<SiStripRawDigi> >* outputPayloadDigisVector = NULL;
    std::vector< edm::DetSet<SiStripRawDigi> >* outputReorderedDigisVector = NULL;
    std::vector< edm::DetSet<SiStripRawDigi> >* outputVirginRawDigisVector = NULL;
    std::set<uint16_t> alreadyMergedFeds;
    for (SpyEventList::const_iterator iMatch = matchingEvents->begin(); iMatch != matchingEvents->end(); ++iMatch) {
      SpyEventPtr event = readSpecificEvent(*iMatch);
      //read the input collections from the event
      const FEDRawDataCollection& inputRawData = *getProduct< FEDRawDataCollection >(event,rawDataTag_);
      const std::vector<uint32_t>& inputTotalEventCounters = *getProduct< std::vector<uint32_t> >(event,totalEventCountersTag_);
      const std::vector<uint32_t>& inputL1ACounters = *getProduct< std::vector<uint32_t> >(event,l1aCountersTag_);
      const std::vector<uint32_t>& inputAPVAddresses = *getProduct< std::vector<uint32_t> >(event,apvAddressesTag_);
      const edm::DetSetVector<SiStripRawDigi>* inputScopeDigis = getProduct< edm::DetSetVector<SiStripRawDigi> >(event,scopeDigisTag_);
      const edm::DetSetVector<SiStripRawDigi>* inputPayloadDigis = getProduct< edm::DetSetVector<SiStripRawDigi> >(event,payloadDigisTag_);
      const edm::DetSetVector<SiStripRawDigi>* inputReorderedDigis = getProduct< edm::DetSetVector<SiStripRawDigi> >(event,reorderedDigisTag_);
      const edm::DetSetVector<SiStripRawDigi>* inputVirginRawDigis = getProduct< edm::DetSetVector<SiStripRawDigi> >(event,virginRawDigisTag_);
      //construct the output vectors if the digis were found and they do not exist
      if (inputScopeDigis && !outputScopeDigisVector) outputScopeDigisVector = new std::vector< edm::DetSet<SiStripRawDigi> >;
      if (inputPayloadDigis && !outputPayloadDigisVector) outputPayloadDigisVector = new std::vector< edm::DetSet<SiStripRawDigi> >;
      if (inputReorderedDigis && !outputReorderedDigisVector) outputReorderedDigisVector = new std::vector< edm::DetSet<SiStripRawDigi> >;
      if (inputVirginRawDigis && !outputVirginRawDigisVector) outputVirginRawDigisVector = new std::vector< edm::DetSet<SiStripRawDigi> >;
      //find matching FEDs
      std::set<uint16_t> matchingFeds;
      findMatchingFeds(eventId,apvAddress,inputTotalEventCounters,inputL1ACounters,inputAPVAddresses,matchingFeds);
      LogInfo(mlLabel_) << "Spy event " << iMatch->event() << " has " << matchingFeds.size() << " matching FEDs";
      std::ostringstream ss;
      ss << "Matching FEDs for event " << iMatch->event() << ": ";
      for (std::set<uint16_t>::const_iterator iFedId = matchingFeds.begin(); iFedId != matchingFeds.end(); ++iFedId) {
        ss << *iFedId << " ";
      }
      LogDebug(mlLabel_) << ss.str();
      //check there are no duplicats
      std::vector<uint16_t> duplicateFeds( std::min(alreadyMergedFeds.size(),matchingFeds.size()) );
      std::vector<uint16_t>::iterator duplicatesBegin = duplicateFeds.begin();
      std::vector<uint16_t>::iterator duplicatesEnd = std::set_intersection(alreadyMergedFeds.begin(),alreadyMergedFeds.end(),
                                                                            matchingFeds.begin(),matchingFeds.end(),
                                                                            duplicatesBegin);
      if ( (duplicatesEnd-duplicatesBegin) != 0 ) {
        std::ostringstream ss;
        ss << "Found a match for FEDs ";
        for (std::vector<uint16_t>::const_iterator iDup = duplicatesBegin; iDup != duplicatesEnd; ++iDup) {
          ss << *iDup << " ";
        }
        ss << ". Output SetSetVectors will be unusable!";
        LogError(mlLabel_) << ss.str();
      }
      //merge the matching data
      mergeMatchingData(matchingFeds,inputRawData,inputTotalEventCounters,inputL1ACounters,inputAPVAddresses,
                        inputScopeDigis,inputPayloadDigis,inputReorderedDigis,inputVirginRawDigis,
                        outputRawData,outputTotalEventCounters,outputL1ACounters,outputAPVAddresses,
                        outputScopeDigisVector,outputPayloadDigisVector,outputReorderedDigisVector,outputVirginRawDigisVector,
                        cabling);
      alreadyMergedFeds.insert(matchingFeds.begin(),matchingFeds.end());
    }
    SpyDataCollections collections(outputRawData,outputTotalEventCounters,outputL1ACounters,outputAPVAddresses,
                                   outputScopeDigisVector,outputPayloadDigisVector,outputReorderedDigisVector,outputVirginRawDigisVector);
    collectionsToCreate = collections;
  }
  
  void SpyEventMatcher::findMatchingFeds(const uint32_t eventId, const uint8_t apvAddress,
                                         std::vector<uint32_t> totalEventCounters,
                                         std::vector<uint32_t> l1aCounters,
                                         std::vector<uint32_t> apvAddresses,
                                         std::set<uint16_t>& matchingFeds)
  {
    //loop over all FEDs. Maps should have same content and be in order so, avoid searches by iterating (and checking keys match)
    std::vector<uint32_t>::const_iterator iTotalEventCount = totalEventCounters.begin();
    std::vector<uint32_t>::const_iterator iL1ACount = l1aCounters.begin();
    std::vector<uint32_t>::const_iterator iAPVAddress = apvAddresses.begin();
    for (;
         ( (iTotalEventCount != totalEventCounters.end()) && (iL1ACount != l1aCounters.end()) && (iAPVAddress != apvAddresses.end()) );
         (++iTotalEventCount, ++iL1ACount, ++iAPVAddress)
        ){
      if ( (eventId > *iTotalEventCount) && (eventId <= *iL1ACount+1) && (*iAPVAddress == apvAddress) ) {
        matchingFeds.insert(matchingFeds.end(),*iTotalEventCount);
      }
    }
  }
  
  void SpyEventMatcher::mergeMatchingData(const std::set<uint16_t>& matchingFeds,
                                        const FEDRawDataCollection& inputRawData,
                                        const std::vector<uint32_t>& inputTotalEventCounters,
                                        const std::vector<uint32_t>& inputL1ACounters,
                                        const std::vector<uint32_t>& inputAPVAddresses,
                                        const edm::DetSetVector<SiStripRawDigi>* inputScopeDigis,
                                        const edm::DetSetVector<SiStripRawDigi>* inputPayloadDigis,
                                        const edm::DetSetVector<SiStripRawDigi>* inputReorderedDigis,
                                        const edm::DetSetVector<SiStripRawDigi>* inputVirginRawDigis,
                                        FEDRawDataCollection& outputRawData,
                                        std::vector<uint32_t>& outputTotalEventCounters,
                                        std::vector<uint32_t>& outputL1ACounters,
                                        std::vector<uint32_t>& outputAPVAddresses,
                                        std::vector< edm::DetSet<SiStripRawDigi> >* outputScopeDigisVector,
                                        std::vector< edm::DetSet<SiStripRawDigi> >* outputPayloadDigisVector,
                                        std::vector< edm::DetSet<SiStripRawDigi> >* outputReorderedDigisVector,
                                        std::vector< edm::DetSet<SiStripRawDigi> >* outputVirginRawDigisVector,
                                        const SiStripFedCabling& cabling)
  {
    //reserve space in vectors
    if (inputScopeDigis) {
      outputScopeDigisVector->reserve(outputScopeDigisVector->size()+matchingFeds.size()*FEDCH_PER_FED); //maximum number of channels on matching FEDs
    }
    if (inputPayloadDigis) {
      outputPayloadDigisVector->reserve(outputPayloadDigisVector->size()+matchingFeds.size()*FEDCH_PER_FED);
    }
    if (inputReorderedDigis) {
      outputReorderedDigisVector->reserve(outputReorderedDigisVector->size()+matchingFeds.size()*FEDCH_PER_FED);
    }
    if (inputVirginRawDigis) {
      outputVirginRawDigisVector->reserve(outputVirginRawDigisVector->size()+matchingFeds.size()*FEDCH_PER_FED/2); //maximum number of dets on matching FEDs
    }
    //copy the data into output collections
    std::set<uint32_t> usedDetIds;
    for (std::set<uint16_t>::const_iterator iFedId = matchingFeds.begin(); iFedId != matchingFeds.end(); ++iFedId) {
      const uint32_t fedId = *iFedId;
      outputRawData.FEDData(fedId) = inputRawData.FEDData(fedId);
      outputTotalEventCounters[fedId] = inputTotalEventCounters[fedId];
      outputL1ACounters[fedId] = inputL1ACounters[fedId];
      outputAPVAddresses[fedId] = inputAPVAddresses[fedId];
      for (uint8_t chan = 0; chan < FEDCH_PER_FED; ++chan) {
        uint32_t fedIndex = SiStripFedKey::fedIndex(fedId,chan);
        if (inputScopeDigis) {
          edm::DetSetVector<SiStripRawDigi>::const_iterator iScopeDigis = inputScopeDigis->find(fedIndex);
          if (iScopeDigis != inputScopeDigis->end()) {
            outputScopeDigisVector->push_back(*iScopeDigis);
          }
        }
        if (inputPayloadDigis) {
          edm::DetSetVector<SiStripRawDigi>::const_iterator iPayloadDigis = inputPayloadDigis->find(fedIndex);
          if (iPayloadDigis != inputPayloadDigis->end()) {
            outputPayloadDigisVector->push_back(*iPayloadDigis);
          }
        }
        if (inputReorderedDigis) {
          edm::DetSetVector<SiStripRawDigi>::const_iterator iReorderedDigis = inputReorderedDigis->find(fedIndex);
          if (iReorderedDigis != inputReorderedDigis->end()) {
            outputReorderedDigisVector->push_back(*iReorderedDigis);
          }
        }
      }
      if (inputVirginRawDigis) {
        std::set<uint32_t> fedDetIds;
        const std::vector<FedChannelConnection>& conns = cabling.connections(fedId);
        for (std::vector<FedChannelConnection>::const_iterator iConn = conns.begin(); iConn != conns.end(); ++iConn) {
          if (!iConn->isConnected()) continue;
          const uint32_t detId = iConn->detId();
          if (usedDetIds.find(detId) != usedDetIds.end()) {
            LogError(mlLabel_) << "Duplicate DetID found " << detId << " skipping data for this Det from FED " << fedId;
            continue;
          }
          fedDetIds.insert(iConn->detId());
        }
        usedDetIds.insert(fedDetIds.begin(),fedDetIds.end());
        for (std::set<uint32_t>::const_iterator iDetId = fedDetIds.begin(); iDetId != fedDetIds.end(); ++iDetId) {
          edm::DetSetVector<SiStripRawDigi>::const_iterator iVirginRawDigis = inputVirginRawDigis->find(*iDetId);
          if (iVirginRawDigis != inputVirginRawDigis->end()) {
            outputVirginRawDigisVector->push_back(*iVirginRawDigis);
          }
        }
      }
    }
  }
  
  SpyEventMatcher::SpyDataCollections::SpyDataCollections(FEDRawDataCollection& theRawData,
                                                        std::vector<uint32_t>& theTotalEventCounters,
                                                        std::vector<uint32_t>& theL1ACounters,
                                                        std::vector<uint32_t>& theAPVAddresses,
                                                        std::vector< edm::DetSet<SiStripRawDigi> >* theScopeDigisVector,
                                                        std::vector< edm::DetSet<SiStripRawDigi> >* thePayloadDigisVector,
                                                        std::vector< edm::DetSet<SiStripRawDigi> >* theReorderedDigisVector,
                                                        std::vector< edm::DetSet<SiStripRawDigi> >* theVirginRawDigisVector)
    : rawData(new FEDRawDataCollection),
      totalEventCounters(new std::vector<uint32_t>),
      l1aCounters(new std::vector<uint32_t>),
      apvAddresses(new std::vector<uint32_t>),
      scopeDigis(theScopeDigisVector ? new edm::DetSetVector<SiStripRawDigi>(*theScopeDigisVector) : NULL),
      payloadDigis(thePayloadDigisVector ? new edm::DetSetVector<SiStripRawDigi>(*thePayloadDigisVector) : NULL),
      reorderedDigis(theReorderedDigisVector ? new edm::DetSetVector<SiStripRawDigi>(*theReorderedDigisVector) : NULL),
      virginRawDigis(theVirginRawDigisVector ? new edm::DetSetVector<SiStripRawDigi>(*theVirginRawDigisVector) : NULL)
  {
    rawData->swap(theRawData);
    totalEventCounters->swap(theTotalEventCounters);
    l1aCounters->swap(theL1ACounters);
    apvAddresses->swap(theAPVAddresses);
  }
  
  SpyEventMatcher::SpyDataCollections::SpyDataCollections()
    : rawData(),
      totalEventCounters(),
      l1aCounters(),
      apvAddresses(),
      scopeDigis(),
      payloadDigis(),
      reorderedDigis(),
      virginRawDigis()
  {}
  
  SpyEventMatcher::SpyDataCollections& SpyEventMatcher::SpyDataCollections::operator = (SpyDataCollections original)
  {
    rawData = original.rawData;
    totalEventCounters = original.totalEventCounters;
    l1aCounters = original.l1aCounters;
    apvAddresses = original.apvAddresses;
    scopeDigis = original.scopeDigis;
    payloadDigis = original.payloadDigis;
    virginRawDigis = original.virginRawDigis;
    return *this;
  }
  
}

#endif //SiStripMonitorHardware_BuildEventMatchingCode
