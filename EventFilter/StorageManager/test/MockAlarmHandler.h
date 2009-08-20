// $Id: MockAlarmHandler.h,v 1.5 2009/08/18 09:15:49 mommsen Exp $
/// @file: MockAlarmHandler.h 

#ifndef StorageManager_MockAlarmHandler_h
#define StorageManager_MockAlarmHandler_h

#include <string>
#include <vector>
#include <map>

#include "EventFilter/StorageManager/interface/AlarmHandler.h"


namespace stor {


  class MockAlarmHandler : public AlarmHandler
  {

  public:
    
    MockAlarmHandler() {}
    virtual ~MockAlarmHandler() {}

    typedef std::pair<ALARM_LEVEL,xcept::Exception> Alarms;
    typedef std::multimap<std::string, Alarms> AlarmsList;

    /**
      Raises a sentinel alarm
    */
    void raiseAlarm
    (
      const std::string name,
      const ALARM_LEVEL level,
      xcept::Exception& e
    )
    {
      Alarms alarm = std::make_pair(level, e);
      _alarmsList.insert(std::make_pair(name, alarm));
    }

    /**
      Revokes s sentinel alarm
    */
    void revokeAlarm(const std::string name)
    {
      _alarmsList.erase(name);
    }
 
    bool noAlarmSet()
    {
      return _alarmsList.empty();
    }

    bool getActiveAlarms(const std::string name, std::vector<Alarms>& alarms)
    {
      for (AlarmsList::iterator it = _alarmsList.lower_bound(name),
             itEnd =  _alarmsList.upper_bound(name);
           it != itEnd;
           ++it)
      {
        alarms.push_back(it->second);
      }
      return !alarms.empty();
    }

  private:

    AlarmsList _alarmsList;

  };
  
} // namespace stor

#endif // StorageManager_MockAlarmHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
