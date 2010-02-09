// $Id: RunMonitorCollection.h,v 1.6 2009/08/24 14:31:11 mommsen Exp $
/// @file: RunMonitorCollection.h 

#ifndef StorageManager_RunMonitorCollection_h
#define StorageManager_RunMonitorCollection_h

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
   * $Revision: 1.6 $
   * $Date: 2009/08/24 14:31:11 $
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

      bool operator<(UnwantedEventKey const& other) const;
    };
    struct UnwantedEventValue
    {
      uint32 count;
      bool sentFirstAlarm;

      UnwantedEventValue() :
      count(1), sentFirstAlarm(false) {};
    };
    typedef std::map<UnwantedEventKey, UnwantedEventValue> UnwantedEventsMap;
    UnwantedEventsMap _unwantedEvents;

    void checkForBadEvents();
    void alarmErrorEvents();
    void alarmUnwantedEvents(UnwantedEventsMap::value_type&);

    xdata::UnsignedInteger32 _runNumber;           // The current run number

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
