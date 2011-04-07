// $Id: AlarmHandler.h,v 1.7.8.3 2011/02/28 17:56:15 mommsen Exp $
/// @file: AlarmHandler.h 

#ifndef EventFilter_StorageManager_AlarmHandler_h
#define EventFilter_StorageManager_AlarmHandler_h

#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"

#include "xcept/Exception.h"
#include "xdaq/Application.h"
#include "xdata/InfoSpace.h"


namespace stor {

  /**
   * Helper class to handle sentinel alarming
   *
   * $Author: mommsen $
   * $Revision: 1.7.8.3 $
   * $Date: 2011/02/28 17:56:15 $
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
      Revokes a sentinel alarm
    */
    virtual void revokeAlarm(const std::string name);

    /**
      Revokes all sentinel alarms 
    */
    void clearAllAlarms();


  private:

    bool raiseAlarm
    (
      const std::string name,
      const std::string level,
      xcept::Exception&
    );

    xdaq::Application* app_;
    xdata::InfoSpace* alarmInfoSpace_;

    mutable boost::mutex mutex_;

  };
  
  typedef boost::shared_ptr<AlarmHandler> AlarmHandlerPtr;

} // namespace stor

#endif // EventFilter_StorageManager_AlarmHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
