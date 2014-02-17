// $Id: AlarmHandler.h,v 1.10 2011/11/08 10:48:39 mommsen Exp $
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

  class SharedResources;


  /**
   * Helper class to handle sentinel alarming
   *
   * $Author: mommsen $
   * $Revision: 1.10 $
   * $Date: 2011/11/08 10:48:39 $
   */

  class AlarmHandler
  {

  public:

    enum ALARM_LEVEL { OKAY, WARNING, ERROR, FATAL };
    
    // Constructor for MockAlarmHandler (unit tests)
    AlarmHandler() {};

    // Constructor for SMProxy
    explicit AlarmHandler(xdaq::Application*);

    // Constructor for SM
    AlarmHandler
    (
      xdaq::Application*,
      boost::shared_ptr<SharedResources>
    );

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

    /**
     * Add a Failed state-machine event to the command queue
     */
    virtual void moveToFailedState( xcept::Exception& );

    /**
       Write message to a file in /tmp
       (last resort when everything else fails)
    */
    void localDebug( const std::string& message ) const;

    /**
      Return the application logger
    */
    Logger& getLogger() const
    { return app_->getApplicationLogger(); }


  private:

    bool raiseAlarm
    (
      const std::string name,
      const std::string level,
      xcept::Exception&
    );

    xdaq::Application* app_;
    boost::shared_ptr<SharedResources> sharedResources_;
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
