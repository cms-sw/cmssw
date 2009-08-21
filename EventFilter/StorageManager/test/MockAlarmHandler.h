// $Id: MockAlarmHandler.h,v 1.1 2009/08/20 13:48:46 mommsen Exp $
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

    bool getActiveAlarms(const std::string& name, std::vector<Alarms>& alarms)
    {
      std::pair<AlarmsList::iterator, AlarmsList::iterator> range;
      range = _alarmsList.equal_range(name);
      for (AlarmsList::iterator it = range.first;
           it != range.second;
           ++it)
      {
        alarms.push_back(it->second);
      }
      return !alarms.empty();
    }

    void printActiveAlarms(const std::string& name)
    {
      std::cout << "\nActive alarms for " << name << std::endl;
      
      std::pair<AlarmsList::iterator, AlarmsList::iterator> range;
      range = _alarmsList.equal_range(name);
      for (AlarmsList::iterator it = range.first;
           it != range.second;
           ++it)
      {
        std::cout << "   " << it->second.first << "\t" << it->second.second.message() << std::endl;
      }
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
