#define SiStripMonitorHardware_BuildEventMatchingCode
#ifdef SiStripMonitorHardware_BuildEventMatchingCode

#ifndef DQM_SiStripMonitorHardware_SiStripSpyEventMatcher_H
#define DQM_SiStripMonitorHardware_SiStripSpyEventMatcher_H

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/cstdint.hpp"
#include "boost/shared_ptr.hpp"
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
        std::unique_ptr< FEDRawDataCollection > rawData;
        std::unique_ptr< std::vector<uint32_t> > totalEventCounters;
        std::unique_ptr< std::vector<uint32_t> > l1aCounters;
        std::unique_ptr< std::vector<uint32_t> > apvAddresses;
        std::unique_ptr< edm::DetSetVector<SiStripRawDigi> > scopeDigis;
        std::unique_ptr< edm::DetSetVector<SiStripRawDigi> > payloadDigis;
        std::unique_ptr< edm::DetSetVector<SiStripRawDigi> > reorderedDigis;
        std::unique_ptr< edm::DetSetVector<SiStripRawDigi> > virginRawDigis;
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
      };
      struct MatchingOutput
      {
        explicit MatchingOutput(FEDRawDataCollection& outputRawData);
        FEDRawDataCollection& outputRawData_;
        std::vector<uint32_t> outputTotalEventCounters_;
        std::vector<uint32_t> outputL1ACounters_;
        std::vector<uint32_t> outputAPVAddresses_;
        boost::shared_ptr< std::vector< edm::DetSet<SiStripRawDigi> > > outputScopeDigisVector_;
        boost::shared_ptr< std::vector< edm::DetSet<SiStripRawDigi> > > outputPayloadDigisVector_;
        boost::shared_ptr< std::vector< edm::DetSet<SiStripRawDigi> > > outputReorderedDigisVector_;
        boost::shared_ptr< std::vector< edm::DetSet<SiStripRawDigi> > > outputVirginRawDigisVector_;
        std::set<uint16_t> alreadyMergedFeds_;
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
      //helper for getMatchedCollections() 
      void getCollections(const edm::EventPrincipal& event, const uint32_t eventId,
                          const uint8_t apvAddress, const SiStripFedCabling& cabling,
                          MatchingOutput& matchingOutput);

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
      typedef std::vector<uint32_t> Counters;
      //class to wrap counters that can take ownership of them or not.
      //It behaves like the counters themself but, it actualy holds a pointer to them and dletes them if necessary
      class CountersWrapper
      {
        public:
          CountersWrapper(const Counters* theCounters);
          CountersWrapper(Counters* theCounters, const bool takeOwnership);
          ~CountersWrapper();
          const Counters::value_type operator [] (const size_t i) const { return (*pConst)[i]; };
          const Counters::value_type at(const size_t i) const { return pConst->at(i); };
          Counters::const_iterator begin() const { return pConst->begin(); }
          Counters::const_iterator end() const { return pConst->end(); }
        private:
          const Counters* pConst;
          Counters* p;
          bool deleteP;
      };
      typedef boost::shared_ptr<CountersWrapper> CountersPtr;
      
      //source for spy events
      typedef edm::VectorInputSource Source;
      //reference counted pointer to an event
            
      std::unique_ptr<Source> constructSource(const edm::ParameterSet& sourceConfig);
      void addNextEventToMap(const edm::EventPrincipal& nextSpyEvent);
      template <class T> static const T* getProduct(const edm::EventPrincipal& event, const edm::InputTag& tag);
      static CountersPtr getCounters(const edm::EventPrincipal& event, const edm::InputTag& tag, const bool mapKeyIsByFedID = true);
      void operator()(const edm::EventPrincipal& event);
      static void findMatchingFeds(const uint32_t eventId, const uint8_t apvAddress,
                                   CountersPtr totalEventCounters,
                                   CountersPtr l1aCounters,
                                   CountersPtr apvAddresses,
                                   std::set<uint16_t>& matchingFeds);
      static void mergeMatchingData(const std::set<uint16_t>& matchingFeds,
                                    const FEDRawDataCollection& inputRawData,
                                    CountersPtr inputTotalEventCounters,
                                    CountersPtr inputL1ACounters,
                                    CountersPtr inputAPVAddresses,
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
      uint32_t counterDiffMax_;
      std::shared_ptr<edm::ProductRegistry> productRegistry_;
      std::unique_ptr<edm::VectorInputSource> const source_;
      std::unique_ptr<edm::ProcessConfiguration> processConfiguration_;
      std::unique_ptr<edm::EventPrincipal> eventPrincipal_;
      static const char* mlLabel_;
  };
  
  template <class T> const T* SpyEventMatcher::getProduct(const edm::EventPrincipal& event, const edm::InputTag& tag)
  {
    LogDebug(mlLabel_) << "Retrieving product " << tag;
    // Note: The third argument to getProductByTag can be a nullptr
    // as long as unscheduled execution of an EDProducer cannot occur
    // as a result of this function call (and with the current implementation
    // of SpyEventMatcher unscheduled execution never happens).
    const std::shared_ptr< const edm::Wrapper<T> > productWrapper = edm::getProductByTag<T>(event,tag,nullptr);
    if (productWrapper) {
      return productWrapper->product();
    } else {
      return nullptr;
    }
  }
  
  inline bool SpyEventMatcher::EventKey::operator < (const SpyEventMatcher::EventKey& rhs) const
  {
    return ( (this->eventId_<rhs.eventId_) ? true : ( ((this->eventId_==rhs.eventId_) && (this->apvAddress_<rhs.apvAddress_)) ? true : false) );
  }
  
}

#endif //ndef DQM_SiStripMonitorHardware_SiStripSpyEventMatcher_H

#endif //SiStripMonitorHardware_BuildEventMatchingCode
