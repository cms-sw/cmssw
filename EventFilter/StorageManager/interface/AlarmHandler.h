// $Id: AlarmHandler.h,v 1.5 2009/08/18 09:15:49 mommsen Exp $
/// @file: AlarmHandler.h 

#ifndef StorageManager_AlarmHandler_h
#define StorageManager_AlarmHandler_h

#include <string>

#include "xcept/Exception.h"
#include "xdaq/Application.h"
#include "xdata/InfoSpace.h"


namespace stor {

  /**
   * Helper class to handle sentinel alarming
   *
   * $Author: mommsen $
   * $Revision: 1.3 $
   * $Date: 2009/07/20 13:06:11 $
   */

  class AlarmHandler
  {

  public:

    enum ALARM_LEVEL { OKAY, WARNING, ERROR, FATAL };
    
    AlarmHandler() {};
    explicit AlarmHandler(xdaq::Application*);

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
      Revokes s sentinel alarm
    */
    virtual void revokeAlarm(const std::string name);
 

  private:

    void raiseAlarm
    (
      const std::string name,
      const std::string level,
      xcept::Exception&
    );

    xdaq::Application* _app;
    xdata::InfoSpace* _alarmInfoSpace;

  };
  
} // namespace stor

#endif // StorageManager_AlarmHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
