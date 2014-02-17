// $Id: RunMonitorCollection.h,v 1.14 2011/11/08 10:48:40 mommsen Exp $
/// @file: RunMonitorCollection.h 

#ifndef EventFilter_StorageManager_RunMonitorCollection_h
#define EventFilter_StorageManager_RunMonitorCollection_h

#include <boost/thread/mutex.hpp>

#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities related to events received
   * in the current run
   *
   * $Author: mommsen $
   * $Revision: 1.14 $
   * $Date: 2011/11/08 10:48:40 $
   */
  
  class RunMonitorCollection : public MonitorCollection
  {
  public:

    RunMonitorCollection
    (
      const utils::Duration_t& updateInterval,
      SharedResourcesPtr
    );

    void configureAlarms(AlarmParams const&);

    const MonitoredQuantity& getEventIDsReceivedMQ() const {
      return eventIDsReceived_;
    }
    MonitoredQuantity& getEventIDsReceivedMQ() {
      return eventIDsReceived_;
    }

    const MonitoredQuantity& getErrorEventIDsReceivedMQ() const {
      return errorEventIDsReceived_;
    }
    MonitoredQuantity& getErrorEventIDsReceivedMQ() {
      return errorEventIDsReceived_;
    }

    const MonitoredQuantity& getUnwantedEventIDsReceivedMQ() const {
      return unwantedEventIDsReceived_;
    }
    MonitoredQuantity& getUnwantedEventIDsReceivedMQ() {
      return unwantedEventIDsReceived_;
    }

    const MonitoredQuantity& getRunNumbersSeenMQ() const {
      return runNumbersSeen_;
    }
    MonitoredQuantity& getRunNumbersSeenMQ() {
      return runNumbersSeen_;
    }

    const MonitoredQuantity& getLumiSectionsSeenMQ() const {
      return lumiSectionsSeen_;
    }
    MonitoredQuantity& getLumiSectionsSeenMQ() {
      return lumiSectionsSeen_;
    }

    const MonitoredQuantity& getEoLSSeenMQ() const {
      return eolsSeen_;
    }
    MonitoredQuantity& getEoLSSeenMQ() {
      return eolsSeen_;
    }

    void addUnwantedEvent(const I2OChain&);


  private:

    //Prevent copying of the RunMonitorCollection
    RunMonitorCollection(RunMonitorCollection const&);
    RunMonitorCollection& operator=(RunMonitorCollection const&);

    MonitoredQuantity eventIDsReceived_;
    MonitoredQuantity errorEventIDsReceived_;
    MonitoredQuantity unwantedEventIDsReceived_;
    MonitoredQuantity runNumbersSeen_;  // Does this make sense?
    MonitoredQuantity lumiSectionsSeen_;
    MonitoredQuantity eolsSeen_;

    SharedResourcesPtr sharedResources_;

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    struct UnwantedEvent
    {
      uint32_t count;
      uint32_t previousCount;
      std::string alarmName;
      uint32_t hltTriggerCount;
      std::vector<unsigned char> bitList;

      UnwantedEvent(const I2OChain&);

      static uint32_t nextId;
    };
    typedef std::map<uint32_t, UnwantedEvent> UnwantedEventsMap;
    UnwantedEventsMap unwantedEventsMap_;
    mutable boost::mutex unwantedEventMapLock_;

    void checkForBadEvents();
    void alarmErrorEvents();
    void alarmUnwantedEvents(UnwantedEventsMap::value_type&);

    xdata::UnsignedInteger32 runNumber_;       // The current run number
    xdata::UnsignedInteger32 dataEvents_;      // Number of data events received
    xdata::UnsignedInteger32 errorEvents_;     // Number of error events received
    xdata::UnsignedInteger32 unwantedEvents_;  // Number of events not consumed

    AlarmParams alarmParams_;
  };
  
} // namespace stor

#endif // EventFilter_StorageManager_RunMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
