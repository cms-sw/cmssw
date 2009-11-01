// $Id: AlarmHandler.h,v 1.4.2.2 2009/09/25 09:57:42 mommsen Exp $
/// @file: AlarmHandler.h 

#ifndef StorageManager_AlarmHandler_h
#define StorageManager_AlarmHandler_h

#include <string>

#include "boost/thread/mutex.hpp"

#include "xcept/Exception.h"
#include "xdaq/Application.h"
#include "xdata/InfoSpace.h"


namespace stor {

  /**
   * Helper class to handle sentinel alarming
   *
   * $Author: mommsen $
   * $Revision: 1.4.2.2 $
   * $Date: 2009/09/25 09:57:42 $
   */

  class AlarmHandler
  {

  public:

    enum ALARM_LEVEL { OKAY, WARNING, ERROR, FATAL };
    
    AlarmHandler() {};
    explicit AlarmHandler(xdaq::Application*);

    virtual ~AlarmHandler() {};

    /**
      Notifies the sentinel
    */
    virtual void notifySentinel
    (
      const ALARM_LEVEL,
      xcept::Exception&
    );

    /**
      Raises a sentinel alarm
    */
    virtual void raiseAlarm
    (
      const std::string name,
      const ALARM_LEVEL,
      xcept::Exception&
    );

    /**
      Revokes a sentinel alarm.
    */
    virtual void revokeAlarm(const std::string name);
 

  private:

    bool raiseAlarm
    (
      const std::string name,
      const std::string level,
      xcept::Exception&
    );

    xdaq::Application* _app;
    xdata::InfoSpace* _alarmInfoSpace;

    mutable boost::mutex _mutex;

  };
  
} // namespace stor

#endif // StorageManager_AlarmHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
