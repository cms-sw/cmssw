// $Id: MockAlarmHandler.h,v 1.6 2011/11/08 10:48:42 mommsen Exp $
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
      Notifies the sentinel
    */
    virtual void notifySentinel
    (
      const ALARM_LEVEL level,
      xcept::Exception& e
    )
    {
      raiseAlarm("SentinelException", level, e);
    }

    /**
      Raises a sentinel alarm
    */
    virtual void raiseAlarm
    (
      const std::string name,
      const ALARM_LEVEL level,
      xcept::Exception& e
    )
    {
      Alarms alarm = std::make_pair(level, e);
      alarmsList_.insert(std::make_pair(name, alarm));
    }

    /**
      Revokes s sentinel alarm
    */
    virtual void revokeAlarm(const std::string name)
    {
      alarmsList_.erase(name);
    }
 
    bool noAlarmSet()
    {
      return alarmsList_.empty();
    }

    bool getActiveAlarms(const std::string& name, std::vector<Alarms>& alarms)
    {
      std::pair<AlarmsList::iterator, AlarmsList::iterator> range;
      range = alarmsList_.equal_range(name);
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
      range = alarmsList_.equal_range(name);
      for (AlarmsList::iterator it = range.first;
           it != range.second;
           ++it)
      {
        std::cout << "   " << it->second.first << "\t" << it->second.second.message() << std::endl;
      }
    }

    void moveToFailedState(xcept::Exception& exception)
    {
      notifySentinel(AlarmHandler::FATAL, exception);
    }

  private:

    AlarmsList alarmsList_;

  };
  
} // namespace stor

#endif // StorageManager_MockAlarmHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
