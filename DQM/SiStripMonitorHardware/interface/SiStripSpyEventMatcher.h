#define SiStripMonitorHardware_BuildEventMatchingCode
#ifdef SiStripMonitorHardware_BuildEventMatchingCode

#ifndef DQM_SiStripMonitorHardware_SiStripSpyEventMatcher_H
#define DQM_SiStripMonitorHardware_SiStripSpyEventMatcher_H

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/cstdint.hpp"
#include <set>
#include <map>
#include <memory>
#include <vector>

//forward declarations
class FEDRawDataCollection;
class SiStripRawDigi;
namespace edm {
  template<class T> class DetSetVector;
  template<class T> class DetSet;
  class EventID;
  class ParameterSet;
}
class SiStripFedCabling;

namespace sistrip {
  
  class SpyEventMatcher
  {
    public:
      class SpyDataCollections
      {
        public:
        std::auto_ptr< FEDRawDataCollection > rawData;
        std::auto_ptr< std::vector<uint32_t> > totalEventCounters;
        std::auto_ptr< std::vector<uint32_t> > l1aCounters;
        std::auto_ptr< std::vector<uint32_t> > apvAddresses;
        std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > scopeDigis;
        std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > payloadDigis;
        std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > reorderedDigis;
        std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > virginRawDigis;
        SpyDataCollections();
        //NB. This will remove all elements in the containers pasted in. It does not copy the data. 
        SpyDataCollections(FEDRawDataCollection& theRawData,
                           std::vector<uint32_t>& theTotalEventCounters,
                           std::vector<uint32_t>& theL1ACounters,
                           std::vector<uint32_t>& theAPVAddresses,
                           std::vector< edm::DetSet<SiStripRawDigi> >* theScopeDigisVector,
                           std::vector< edm::DetSet<SiStripRawDigi> >* thePayloadDigisVector,
                           std::vector< edm::DetSet<SiStripRawDigi> >* theReorderedDigisVector,
                           std::vector< edm::DetSet<SiStripRawDigi> >* theVirginRawDigisVector);
        //does not copy, orginal object looses ownership of collections
        SpyDataCollections& operator = (SpyDataCollections original);
      };
      typedef edm::EventID EventID;
      typedef std::set<EventID> SpyEventList;
      
      SpyEventMatcher(const edm::ParameterSet& config);
      virtual ~SpyEventMatcher();
      //set up the internal map of eventID, to apvAddress
      void initialize();
      //check if there is any data for an event. Returns NULL if not or a pointer to a list of matches if they exist
      const SpyEventList* matchesForEvent(const uint32_t eventId, const uint8_t apvAddress) const;
      //get data for matching FEDs (non-const because reading events from the source modifies it)
      void getMatchedCollections(const uint32_t eventId, const uint8_t apvAddress, const SpyEventList* matchingEvents,
                                 const SiStripFedCabling& cabling, SpyDataCollections& collectionsToCreate);
    private:
      class EventKey
      {
        public:
          EventKey(const uint32_t eventId, const uint8_t apvAddress);
          uint32_t eventId() const { return eventId_; }
          uint8_t apvAddress() const { return apvAddress_; }
          bool operator < (const EventKey& rhs) const;
        private:
          uint32_t eventId_;
          uint8_t  apvAddress_;
      };
      typedef edm::VectorInputSource Source;
      typedef Source::EventPrincipalVectorElement SpyEventPtr;
      
      static std::auto_ptr<Source> constructSource(const edm::ParameterSet& sourceConfig);
      bool addNextEventToMap();
      template <class T> static const T* getProduct(const SpyEventPtr event, const edm::InputTag& tag);
      SpyEventPtr readNextEvent();
      SpyEventPtr readSpecificEvent(const edm::EventID& id);
      static void findMatchingFeds(const uint32_t eventId, const uint8_t apvAddress,
                                   std::vector<uint32_t> totalEventCounters,
                                   std::vector<uint32_t> l1aCounters,
                                   std::vector<uint32_t> apvAddresses,
                                   std::set<uint16_t>& matchingFeds);
      static void mergeMatchingData(const std::set<uint16_t>& matchingFeds,
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
                                    const SiStripFedCabling& cabling);
      
      std::map<EventKey,SpyEventList> eventMatches_;
      edm::InputTag rawDataTag_;
      edm::InputTag totalEventCountersTag_;
      edm::InputTag l1aCountersTag_;
      edm::InputTag apvAddressesTag_;
      edm::InputTag scopeDigisTag_;
      edm::InputTag payloadDigisTag_;
      edm::InputTag reorderedDigisTag_;
      edm::InputTag virginRawDigisTag_;
      std::auto_ptr<Source> source_;
      static const char* mlLabel_;
  };
  
  template <class T> const T* SpyEventMatcher::getProduct(const SpyEventPtr event, const edm::InputTag& tag)
  {
    LogDebug(mlLabel_) << "Retrieving product " << tag;
    const boost::shared_ptr< const edm::Wrapper<T> > productWrapper = edm::getProductByTag<T>(*event,tag);
    if (productWrapper) {
      return productWrapper->product();
    } else {
      return NULL;
    }
  }
  
  inline bool SpyEventMatcher::EventKey::operator < (const SpyEventMatcher::EventKey& rhs) const
  {
    return ( (this->eventId_<rhs.eventId_) ? true : ( ((this->eventId_==rhs.eventId_) && (this->apvAddress_<rhs.apvAddress_)) ? true : false) );
  }
  
}

#endif //ndef DQM_SiStripMonitorHardware_SiStripSpyEventMatcher_H

#endif //SiStripMonitorHardware_BuildEventMatchingCode
