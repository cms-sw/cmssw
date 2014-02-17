//$Id: AlarmHandler.cc,v 1.12 2011/11/08 10:48:40 mommsen Exp $
/// @file: AlarmHandler.cc


#include "sentinel/utils/version.h"
#include "sentinel/utils/Alarm.h"

#include "xcept/tools.h"
#include "xdata/InfoSpaceFactory.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"


namespace stor {

  AlarmHandler::AlarmHandler
  (
    xdaq::Application* app
  ) :
  app_(app)
  {
    try
    {
      alarmInfoSpace_ = xdata::getInfoSpaceFactory()->get("urn:xdaq-sentinel:alarms");
    }
    catch(xdata::exception::Exception)
    {
      // sentinel is not available
      alarmInfoSpace_ = 0;
    }
  }
  

  AlarmHandler::AlarmHandler
  (
    xdaq::Application* app,
    SharedResourcesPtr sr
  ) :
  app_(app),
  sharedResources_(sr)
  {
    try
    {
      alarmInfoSpace_ = xdata::getInfoSpaceFactory()->get("urn:xdaq-sentinel:alarms");
    }
    catch(xdata::exception::Exception)
    {
      // sentinel is not available
      alarmInfoSpace_ = 0;
    }
  }
  
  
  void AlarmHandler::raiseAlarm
  (
    const std::string name,
    const ALARM_LEVEL level,
    xcept::Exception& exception
  )
  {
    
    switch( level )
    {
      case OKAY:
        revokeAlarm(name);
        break;
        
      case WARNING:
        if ( raiseAlarm(name, "warning", exception) )
          LOG4CPLUS_WARN(app_->getApplicationLogger(),
            "Raising warning alarm " << name << ": " << exception.message());
        break;
        
      case ERROR:
        if ( raiseAlarm(name, "error", exception) )
          LOG4CPLUS_ERROR(app_->getApplicationLogger(),
            "Raising error alarm " << name << ": " << exception.message());
        break;
        
      case FATAL:
        if ( raiseAlarm(name, "fatal", exception) )
          LOG4CPLUS_FATAL(app_->getApplicationLogger(),
            "Raising fatal alarm " << name << ": " << exception.message());
        break;
        
      default:
        LOG4CPLUS_WARN(app_->getApplicationLogger(),
          "Unknown alarm level received for " << name << ": " << exception.message());
    }
  }
  
  
  void AlarmHandler::notifySentinel
  (
    const ALARM_LEVEL level,
    xcept::Exception& exception
  )
  {
    
    switch( level )
    {
      case OKAY:
        LOG4CPLUS_INFO(app_->getApplicationLogger(),
          xcept::stdformat_exception_history(exception));
        break;
        
      case WARNING:
        LOG4CPLUS_WARN(app_->getApplicationLogger(),
          xcept::stdformat_exception_history(exception));
        app_->notifyQualified("warning", exception);
        break;
        
      case ERROR:
        LOG4CPLUS_ERROR(app_->getApplicationLogger(),
          xcept::stdformat_exception_history(exception));
        app_->notifyQualified("error", exception);
        break;
        
      case FATAL:
        LOG4CPLUS_FATAL(app_->getApplicationLogger(),
          xcept::stdformat_exception_history(exception));
        app_->notifyQualified("fatal", exception);
        break;
        
      default:
        LOG4CPLUS_WARN(app_->getApplicationLogger(),
          "Unknown alarm level received for exception: " <<
          xcept::stdformat_exception_history(exception));
    }
  }
  
  
  bool AlarmHandler::raiseAlarm
  (
    const std::string name,
    const std::string level,
    xcept::Exception& exception
  )
  {
    
    if (!alarmInfoSpace_) return false;
    
    boost::mutex::scoped_lock sl( mutex_ );
    
    sentinel::utils::Alarm *alarm =
      new sentinel::utils::Alarm(level, exception, app_);
    try
    {
      alarmInfoSpace_->fireItemAvailable(name, alarm);
    }
    catch(xdata::exception::Exception)
    {
      // Alarm is already set or sentinel not available
      return false;
    }
    return true;
  }
  
  
  void AlarmHandler::revokeAlarm
  (
    const std::string name
  )
  {
    if (!alarmInfoSpace_) return;
    
    boost::mutex::scoped_lock sl( mutex_ );
    
    sentinel::utils::Alarm *alarm;
    try
    {
      alarm = dynamic_cast<sentinel::utils::Alarm*>( alarmInfoSpace_->find( name ) );
    }
    catch(xdata::exception::Exception)
    {
      // Alarm has not been set or sentinel not available
      return;
    }
    
    LOG4CPLUS_INFO(app_->getApplicationLogger(), "Revoking alarm " << name);
    
    alarmInfoSpace_->fireItemRevoked(name, app_);
    delete alarm;
  }
  
  
  void AlarmHandler::clearAllAlarms()
  {
    if (!alarmInfoSpace_) return;
    
    boost::mutex::scoped_lock sl( mutex_ );
    
    typedef std::map<std::string, xdata::Serializable*, std::less<std::string> > alarmList;
    alarmList alarms = alarmInfoSpace_->match(".*");
    for (alarmList::const_iterator it = alarms.begin(), itEnd = alarms.end();
         it != itEnd; ++it)
    {
      sentinel::utils::Alarm* alarm = dynamic_cast<sentinel::utils::Alarm*>(it->second);
      alarmInfoSpace_->fireItemRevoked(it->first, app_);
      delete alarm;
    }
  }
  
  
  void AlarmHandler::moveToFailedState( xcept::Exception& exception )
  {
    std::string errorMsg = "Failed to process FAIL exception: "
      + xcept::stdformat_exception_history(exception) + " due to ";

    try
    {
      notifySentinel(AlarmHandler::FATAL, exception);
      sharedResources_->statisticsReporter_->getStateMachineMonitorCollection().setStatusMessage( 
        xcept::stdformat_exception_history(exception)
      );
      EventPtr_t stMachEvent( new Fail() );
      // wait maximum 5 seconds until enqueuing succeeds
      if ( ! sharedResources_->commandQueue_->enqTimedWait( stMachEvent, boost::posix_time::seconds(5) ) )
      {
        XCEPT_DECLARE_NESTED( stor::exception::StateTransition,
          sentinelException, "Failed to enqueue FAIL event", exception );
        notifySentinel(AlarmHandler::FATAL, sentinelException);
      }
    }
    catch(xcept::Exception &e)
    {
      errorMsg += xcept::stdformat_exception_history(e);
      localDebug( errorMsg );
    }
    catch(std::exception &e)
    {
      errorMsg += e.what();
      localDebug( errorMsg );
    }
    catch( ... )
    {
      errorMsg += "an unknown exception.";
      localDebug( errorMsg );
    }
  }
 

  void AlarmHandler::localDebug( const std::string& message ) const
  {
    std::ostringstream fname_oss;
    fname_oss << "/tmp/storage_manager_debug_" << 
      sharedResources_->configuration_->getDiskWritingParams().smInstanceString_ <<
      "_" << getpid();
    const std::string fname = fname_oss.str();
    std::ofstream f( fname.c_str(), std::ios_base::ate | std::ios_base::out | std::ios_base::app );
    if( f.is_open() )
    {
      try
      {
        f << message << std::endl;
        f.close();
      }
      catch(...)
      {}
    }
  }
  
} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
