//$Id: AlarmHandler.cc,v 1.4 2009/09/22 14:54:50 dshpakov Exp $
/// @file: AlarmHandler.cc


#include "sentinel/utils/version.h"
#if SENTINELUTILS_VERSION_MAJOR>1
#include "sentinel/utils/Alarm.h"
#endif

#include "xdata/InfoSpaceFactory.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"


using namespace stor;

AlarmHandler::AlarmHandler(xdaq::Application* app) :
_app(app)
{
#if SENTINELUTILS_VERSION_MAJOR>1
  try
  {
    _alarmInfoSpace = xdata::getInfoSpaceFactory()->get("urn:xdaq-sentinel:alarms");
  }
  catch(xdata::exception::Exception)
  {
    // sentinel is not available
    _alarmInfoSpace = 0;
  }
#endif
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
        LOG4CPLUS_WARN(_app->getApplicationLogger(),
          "Raising warning alarm " << name << ": " << exception.message());
      break;

    case ERROR:
      if ( raiseAlarm(name, "error", exception) )
        LOG4CPLUS_ERROR(_app->getApplicationLogger(),
          "Raising error alarm " << name << ": " << exception.message());
      break;

    case FATAL:
      if ( raiseAlarm(name, "fatal", exception) )
        LOG4CPLUS_FATAL(_app->getApplicationLogger(),
          "Raising fatal alarm " << name << ": " << exception.message());
      break;

    default:
      LOG4CPLUS_WARN(_app->getApplicationLogger(),
        "Unknown alarm level received for " << name << ": " << exception.message());
  }
}

bool AlarmHandler::raiseAlarm
(
  const std::string name,
  const std::string level,
  xcept::Exception& exception
)
{

  if (!_alarmInfoSpace) return false;

  boost::mutex::scoped_lock sl( _mutex );

  #if SENTINELUTILS_VERSION_MAJOR>1
  
  sentinel::utils::Alarm *alarm =
    new sentinel::utils::Alarm(level, exception, _app);
  try
  {
    _alarmInfoSpace->fireItemAvailable(name, alarm);
  }
  catch(xdata::exception::Exception)
  {
    // Alarm is already set or sentinel not available
    return false;
  }
  return true;

  #endif
}


void AlarmHandler::revokeAlarm
(
  const std::string name
)
{
  if (!_alarmInfoSpace) return;

  boost::mutex::scoped_lock sl( _mutex );

  #if SENTINELUTILS_VERSION_MAJOR>1
  
  sentinel::utils::Alarm *alarm;
  try
  {
    alarm = dynamic_cast<sentinel::utils::Alarm*>( _alarmInfoSpace->find( name ) );
  }
  catch(xdata::exception::Exception)
  {
    // Alarm has not been set or sentinel not available
    return;
  }

  LOG4CPLUS_INFO(_app->getApplicationLogger(), "Revoking alarm " << name);
  
  _alarmInfoSpace->fireItemRevoked(name, _app);
  delete alarm;

  #endif

}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
