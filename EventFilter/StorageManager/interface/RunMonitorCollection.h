// $Id: RunMonitorCollection.h,v 1.7 2010/02/09 14:56:18 mommsen Exp $
/// @file: RunMonitorCollection.h 

#ifndef StorageManager_RunMonitorCollection_h
#define StorageManager_RunMonitorCollection_h

#include <boost/thread/mutex.hpp>

#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities related to events received
   * in the current run
   *
   * $Author: mommsen $
   * $Revision: 1.7 $
   * $Date: 2010/02/09 14:56:18 $
   */
  
  class RunMonitorCollection : public MonitorCollection
  {
  public:

    RunMonitorCollection
    (
      const utils::duration_t& updateInterval,
      boost::shared_ptr<AlarmHandler>
    );

    void configureAlarms(AlarmParams const&);

    const MonitoredQuantity& getEventIDsReceivedMQ() const {
      return _eventIDsReceived;
    }
    MonitoredQuantity& getEventIDsReceivedMQ() {
      return _eventIDsReceived;
    }

    const MonitoredQuantity& getErrorEventIDsReceivedMQ() const {
      return _errorEventIDsReceived;
    }
    MonitoredQuantity& getErrorEventIDsReceivedMQ() {
      return _errorEventIDsReceived;
    }

    const MonitoredQuantity& getUnwantedEventIDsReceivedMQ() const {
      return _unwantedEventIDsReceived;
    }
    MonitoredQuantity& getUnwantedEventIDsReceivedMQ() {
      return _unwantedEventIDsReceived;
    }

    const MonitoredQuantity& getRunNumbersSeenMQ() const {
      return _runNumbersSeen;
    }
    MonitoredQuantity& getRunNumbersSeenMQ() {
      return _runNumbersSeen;
    }

    const MonitoredQuantity& getLumiSectionsSeenMQ() const {
      return _lumiSectionsSeen;
    }
    MonitoredQuantity& getLumiSectionsSeenMQ() {
      return _lumiSectionsSeen;
    }

    const MonitoredQuantity& getEoLSSeenMQ() const {
      return _eolsSeen;
    }
    MonitoredQuantity& getEoLSSeenMQ() {
      return _eolsSeen;
    }

    void addUnwantedEvent(const I2OChain&);


  private:

    //Prevent copying of the RunMonitorCollection
    RunMonitorCollection(RunMonitorCollection const&);
    RunMonitorCollection& operator=(RunMonitorCollection const&);

    MonitoredQuantity _eventIDsReceived;
    MonitoredQuantity _errorEventIDsReceived;
    MonitoredQuantity _unwantedEventIDsReceived;
    MonitoredQuantity _runNumbersSeen;  // Does this make sense?
    MonitoredQuantity _lumiSectionsSeen;
    MonitoredQuantity _eolsSeen;

    boost::shared_ptr<AlarmHandler> _alarmHandler;

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    struct UnwantedEventKey
    {
      uint32 outputModuleId;
      uint32 hltTriggerCount;
      std::vector<unsigned char> bitList;

      UnwantedEventKey(const I2OChain&);
      bool operator<(UnwantedEventKey const& other) const;
    };
    struct UnwantedEventValue
    {
      uint32 count;
      uint32 previousCount;
      std::string alarmName;

      UnwantedEventValue();

      static uint32 nextId;
    };
    typedef std::map<UnwantedEventKey, UnwantedEventValue> UnwantedEventsMap;
    UnwantedEventsMap _unwantedEventsMap;
    mutable boost::mutex _unwantedEventMapLock;

    void checkForBadEvents();
    void alarmErrorEvents();
    void alarmUnwantedEvents(UnwantedEventsMap::value_type&);

    xdata::UnsignedInteger32 _runNumber;       // The current run number
    xdata::UnsignedInteger32 _unwantedEvents;  // Number of events not consumed

    AlarmParams _alarmParams;
  };
  
} // namespace stor

#endif // StorageManager_RunMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
