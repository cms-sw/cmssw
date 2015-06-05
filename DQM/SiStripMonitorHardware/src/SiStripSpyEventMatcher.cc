#include "DQM/SiStripMonitorHardware/interface/SiStripSpyEventMatcher.h"
#ifdef SiStripMonitorHardware_BuildEventMatchingCode

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Sources/interface/VectorInputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"
#include "boost/bind.hpp"
#include <algorithm>
#include <limits>
#include <memory>

using edm::LogInfo;
using edm::LogWarning;
using edm::LogError;

namespace sistrip {
  
  const char* SpyEventMatcher::mlLabel_ = "SpyEventMatcher";
  
  SpyEventMatcher::EventKey::EventKey(const uint32_t eventId, const uint8_t apvAddress)
    : eventId_(eventId), apvAddress_(apvAddress) {}
  
  SpyEventMatcher::MatchingOutput::MatchingOutput(FEDRawDataCollection& outputRawData) :
                           outputRawData_(outputRawData),
                           outputTotalEventCounters_(sistrip::FED_ID_MAX+1),
                           outputL1ACounters_(sistrip::FED_ID_MAX+1),
                           outputAPVAddresses_(sistrip::FED_ID_MAX+1) {}

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
      counterDiffMax_(config.getParameter<uint32_t>("CounterDiffMaxAllowed")),
      productRegistry_(new edm::SignallingProductRegistry),
      source_(constructSource(config.getParameter<edm::ParameterSet>("SpySource"))),
      processConfiguration_(new edm::ProcessConfiguration(std::string("@MIXING"), edm::getReleaseVersion(), edm::getPassID())),
      eventPrincipal_()
  {
    // Use the empty parameter set for the parameter set ID of our "@MIXING" process.
    processConfiguration_->setParameterSetID(edm::ParameterSet::emptyParameterSetID());
    productRegistry_->setFrozen();

    eventPrincipal_.reset(new edm::EventPrincipal(source_->productRegistry(),
                                                  std::make_shared<edm::BranchIDListHelper>(),
                                                  std::make_shared<edm::ThinnedAssociationsHelper>(),
                                                  *processConfiguration_,
                                                  nullptr));
  }
  
  std::unique_ptr<SpyEventMatcher::Source> SpyEventMatcher::constructSource(const edm::ParameterSet& sourceConfig)
  {
    const edm::VectorInputSourceFactory* sourceFactory = edm::VectorInputSourceFactory::get();
    edm::VectorInputSourceDescription description(productRegistry_, edm::PreallocationConfiguration());
    return sourceFactory->makeVectorInputSource(sourceConfig, description);
  }

  void SpyEventMatcher::initialize()
  {
    size_t fileNameHash = 0U;
    //add spy events to the map until there are none left
    source_->loopOverEvents(*eventPrincipal_,fileNameHash,std::numeric_limits<size_t>::max(),boost::bind(&SpyEventMatcher::addNextEventToMap,this,_1));
    //debug
    std::ostringstream ss;
    ss << "Events with possible matches (eventID,apvAddress): ";
    for (std::map<EventKey,SpyEventList>::const_iterator iSpyEvent = eventMatches_.begin(); iSpyEvent != eventMatches_.end(); ++iSpyEvent) {
      ss << "(" << iSpyEvent->first.eventId() << "," << uint16_t(iSpyEvent->first.apvAddress()) << ") ";
    }
    LogDebug(mlLabel_) << ss.str();
  }
  
  void SpyEventMatcher::addNextEventToMap(const edm::EventPrincipal& nextSpyEvent)
  {
    edm::EventID spyEventId = nextSpyEvent.id();

    CountersPtr totalEventCounters = getCounters(nextSpyEvent,totalEventCountersTag_);
    CountersPtr l1aCounters = getCounters(nextSpyEvent,l1aCountersTag_);
    CountersPtr apvAddresses = getCounters(nextSpyEvent,apvAddressesTag_,false);
    //loop over all FEDs. Maps should have same content and be in order so, avoid searches by iterating (and checking keys match)
    //add all possible event keys to the map
    std::vector<uint32_t>::const_iterator iTotalEventCount = totalEventCounters->begin();
    std::vector<uint32_t>::const_iterator iL1ACount = l1aCounters->begin();
    std::vector<uint32_t>::const_iterator iAPVAddress = apvAddresses->begin();
    //for debug
    std::map<EventKey,uint16_t> fedCounts;
    unsigned int fedid = 0;
    for (;
         ( (iTotalEventCount != totalEventCounters->end()) && (iL1ACount != l1aCounters->end()) && (iAPVAddress != apvAddresses->end()) ); 
         (++iTotalEventCount, ++iL1ACount, ++iAPVAddress, ++fedid)
        ){
      if (*iAPVAddress == 0) {
        continue;
      }

      if ( ((*iTotalEventCount) > (*iL1ACount) ) || 
	   ((*iL1ACount)-(*iTotalEventCount) > counterDiffMax_) 
	   ) {
	LogWarning(mlLabel_) << "Spy event " << spyEventId.event() 
			     << " error in counter values for FED " << fedid
			     << ", totCount = " << *iTotalEventCount
			     << ", L1Acount = " << *iL1ACount
			     << std::endl;

	continue;
      }

      for (uint32_t eventId = (*iTotalEventCount)+1; eventId <= (*iL1ACount)+1; ++eventId) {
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

  void SpyEventMatcher::getCollections(const edm::EventPrincipal& event, const uint32_t eventId,
                                       const uint8_t apvAddress, const SiStripFedCabling& cabling,
                                       MatchingOutput& mo) {

      //read the input collections from the event
      const FEDRawDataCollection* inputRawDataPtr = getProduct< FEDRawDataCollection >(event,rawDataTag_);
      if (!inputRawDataPtr) {
        throw cms::Exception(mlLabel_) << "Failed to get raw spy data with tag " << rawDataTag_ << " from spy event";
      }
      const FEDRawDataCollection& inputRawData = *inputRawDataPtr;
      CountersPtr inputTotalEventCounters = getCounters(event,totalEventCountersTag_);
      CountersPtr inputL1ACounters = getCounters(event,l1aCountersTag_);
      CountersPtr inputAPVAddresses = getCounters(event,apvAddressesTag_,false);
      const edm::DetSetVector<SiStripRawDigi>* inputScopeDigis = getProduct< edm::DetSetVector<SiStripRawDigi> >(event,scopeDigisTag_);
      const edm::DetSetVector<SiStripRawDigi>* inputPayloadDigis = getProduct< edm::DetSetVector<SiStripRawDigi> >(event,payloadDigisTag_);
      const edm::DetSetVector<SiStripRawDigi>* inputReorderedDigis = getProduct< edm::DetSetVector<SiStripRawDigi> >(event,reorderedDigisTag_);
      const edm::DetSetVector<SiStripRawDigi>* inputVirginRawDigis = getProduct< edm::DetSetVector<SiStripRawDigi> >(event,virginRawDigisTag_);
      //construct the output vectors if the digis were found and they do not exist
      if (inputScopeDigis && !mo.outputScopeDigisVector_.get() ) mo.outputScopeDigisVector_.reset(new std::vector< edm::DetSet<SiStripRawDigi> >);
      if (inputPayloadDigis && !mo.outputPayloadDigisVector_.get() ) mo.outputPayloadDigisVector_.reset(new std::vector< edm::DetSet<SiStripRawDigi> >);
      if (inputReorderedDigis && !mo.outputReorderedDigisVector_.get() ) mo.outputReorderedDigisVector_.reset(new std::vector< edm::DetSet<SiStripRawDigi> >);
      if (inputVirginRawDigis && !mo.outputVirginRawDigisVector_.get() ) mo.outputVirginRawDigisVector_.reset(new std::vector< edm::DetSet<SiStripRawDigi> >);
      //find matching FEDs
      std::set<uint16_t> matchingFeds;
      findMatchingFeds(eventId,apvAddress,inputTotalEventCounters,inputL1ACounters,inputAPVAddresses,matchingFeds);
      LogInfo(mlLabel_) << "Spy event " << event.id() << " has " << matchingFeds.size() << " matching FEDs";
      std::ostringstream ss;
      ss << "Matching FEDs for event " << event.id() << ": ";
      for (std::set<uint16_t>::const_iterator iFedId = matchingFeds.begin(); iFedId != matchingFeds.end(); ++iFedId) {
        ss << *iFedId << " ";
      }
      LogDebug(mlLabel_) << ss.str();
      //check there are no duplicates
      std::vector<uint16_t> duplicateFeds( std::min(mo.alreadyMergedFeds_.size(),matchingFeds.size()) );
      std::vector<uint16_t>::iterator duplicatesBegin = duplicateFeds.begin();
      std::vector<uint16_t>::iterator duplicatesEnd = std::set_intersection(mo.alreadyMergedFeds_.begin(),mo.alreadyMergedFeds_.end(),
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
                        mo.outputRawData_,mo.outputTotalEventCounters_,mo.outputL1ACounters_,mo.outputAPVAddresses_,
                        mo.outputScopeDigisVector_.get(),mo.outputPayloadDigisVector_.get(),
                        mo.outputReorderedDigisVector_.get(),mo.outputVirginRawDigisVector_.get(),
                        cabling);
      mo.alreadyMergedFeds_.insert(matchingFeds.begin(),matchingFeds.end());
  }

  void SpyEventMatcher::getMatchedCollections(const uint32_t eventId, const uint8_t apvAddress,
                                              const SpyEventList* matchingEvents, const SiStripFedCabling& cabling,
                                              SpyDataCollections& collectionsToCreate)
  {
    if (!matchingEvents) return;
    size_t fileNameHash = 0U;
    FEDRawDataCollection outputRawData;
    MatchingOutput mo(outputRawData);
    source_->loopSpecified(*eventPrincipal_,fileNameHash,matchingEvents->begin(),matchingEvents->end(),boost::bind(&SpyEventMatcher::getCollections,this,_1,
                                                       eventId,apvAddress,boost::cref(cabling),boost::ref(mo)));
    SpyDataCollections collections(mo.outputRawData_,mo.outputTotalEventCounters_,mo.outputL1ACounters_,mo.outputAPVAddresses_,
                                   mo.outputScopeDigisVector_.get(),mo.outputPayloadDigisVector_.get(),
                                   mo.outputReorderedDigisVector_.get(),mo.outputVirginRawDigisVector_.get());
    collectionsToCreate = collections;
  }
  
  void SpyEventMatcher::findMatchingFeds(const uint32_t eventId, const uint8_t apvAddress,
                                         SpyEventMatcher::CountersPtr totalEventCounters,
                                         SpyEventMatcher::CountersPtr l1aCounters,
                                         SpyEventMatcher::CountersPtr apvAddresses,
                                         std::set<uint16_t>& matchingFeds)
  {
    //loop over all FEDs. Maps should have same content and be in order so, avoid searches by iterating (and checking keys match)
    std::vector<uint32_t>::const_iterator iTotalEventCount = totalEventCounters->begin();
    std::vector<uint32_t>::const_iterator iL1ACount = l1aCounters->begin();
    std::vector<uint32_t>::const_iterator iAPVAddress = apvAddresses->begin();
    for (;
         ( (iTotalEventCount != totalEventCounters->end()) && (iL1ACount != l1aCounters->end()) && (iAPVAddress != apvAddresses->end()) );
         (++iTotalEventCount, ++iL1ACount, ++iAPVAddress)
        ){
      if (*iAPVAddress == 0) {
        continue;
      }
      if ( (eventId > *iTotalEventCount) && (eventId <= (*iL1ACount)+1) && (*iAPVAddress == apvAddress) ) {
        matchingFeds.insert(matchingFeds.end(),iTotalEventCount-totalEventCounters->begin());
      }
    }
  }
  
  void SpyEventMatcher::mergeMatchingData(const std::set<uint16_t>& matchingFeds,
                                          const FEDRawDataCollection& inputRawData,
                                          SpyEventMatcher::CountersPtr inputTotalEventCounters,
                                          SpyEventMatcher::CountersPtr inputL1ACounters,
                                          SpyEventMatcher::CountersPtr inputAPVAddresses,
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
      LogDebug(mlLabel_) << "Copying data for FED " << fedId;
      if (inputRawData.FEDData(fedId).size() && inputRawData.FEDData(fedId).data()) {
        outputRawData.FEDData(fedId) = inputRawData.FEDData(fedId);
      }
      outputTotalEventCounters[fedId] = (*inputTotalEventCounters)[fedId];
      outputL1ACounters[fedId] = (*inputL1ACounters)[fedId];
      outputAPVAddresses[fedId] = (*inputAPVAddresses)[fedId];
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
        auto conns = cabling.fedConnections(fedId);
        for (auto iConn = conns.begin(); iConn != conns.end(); ++iConn) {
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
  
  SpyEventMatcher::CountersPtr SpyEventMatcher::getCounters(const edm::EventPrincipal& event, const edm::InputTag& tag, const bool mapKeyIsByFedID)
  {
    const std::vector<uint32_t>* vectorFromEvent = getProduct< std::vector<uint32_t> >(event,tag);
    if (vectorFromEvent) {
      //vector is from event so, will be deleted when the event is destroyed (and not before)
      return CountersPtr(new CountersWrapper(vectorFromEvent) );
    } else {
      const std::map<uint32_t,uint32_t>* mapFromEvent = getProduct< std::map<uint32_t,uint32_t> >(event,tag);
      if (mapFromEvent) {
        std::vector<uint32_t>* newVector = new std::vector<uint32_t>(FED_ID_MAX+1,0);
        if (mapKeyIsByFedID) {
          for (std::map<uint32_t,uint32_t>::const_iterator iIdValue = mapFromEvent->begin(); iIdValue != mapFromEvent->end(); ++iIdValue) {
            newVector->at(iIdValue->first) = iIdValue->second;
          }
        } else {
          SpyUtilities::fillFEDMajorities(*mapFromEvent,*newVector);
        }
// 	std::cout << " -- Map " << tag << std::endl;
// 	for (uint32_t lIt= 0;
// 	     lIt < newVector->size();
// 	     lIt++) {
// 	  std::cout << lIt << " " << newVector->at(lIt) << std::endl;
// 	}
        //vector was allocated here so, will need to be deleted when finished with
        CountersPtr newCountersPtr( new CountersWrapper(newVector,true) );
        return newCountersPtr;
      } else {
        throw cms::Exception(mlLabel_) << "Unable to get product " << tag << " from spy event";
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
  
  SpyEventMatcher::CountersWrapper::CountersWrapper(const Counters* theCounters)
    : pConst(theCounters),
      p(NULL),
      deleteP(false)
  {
  }
  
  SpyEventMatcher::CountersWrapper::CountersWrapper(Counters* theCounters, const bool takeOwnership)
    : pConst(theCounters),
      p(theCounters),
      deleteP(takeOwnership)
  {

  }
  
  SpyEventMatcher::CountersWrapper::~CountersWrapper()
  {
    if (deleteP) delete p;
  }
  
}

#endif //SiStripMonitorHardware_BuildEventMatchingCode
