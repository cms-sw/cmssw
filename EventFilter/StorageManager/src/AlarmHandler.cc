//$Id: Utils.cc,v 1.6 2009/08/18 09:16:49 mommsen Exp $
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
  _alarmInfoSpace = xdata::getInfoSpaceFactory()->get("urn:xdaq-sentinel:alarms");
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
      LOG4CPLUS_WARN(_app->getApplicationLogger(),
        "Raising warning alarm " << name << ": " << exception.message());
      raiseAlarm(name, "warning", exception);
      break;

    case ERROR:
      LOG4CPLUS_ERROR(_app->getApplicationLogger(),
        "Raising error alarm " << name << ": " << exception.message());
      raiseAlarm(name, "error", exception);
      break;

    case FATAL:
      LOG4CPLUS_FATAL(_app->getApplicationLogger(),
        "Raising fatal alarm " << name << ": " << exception.message());
      raiseAlarm(name, "fatal", exception);
      break;

    default:
      LOG4CPLUS_WARN(_app->getApplicationLogger(),
        "Unknown alarm level received for " << name << ": " << exception.message());
  }
}

void AlarmHandler::raiseAlarm
(
  const std::string name,
  const std::string level,
  xcept::Exception& exception
)
{
  #if SENTINELUTILS_VERSION_MAJOR>1
  
  sentinel::utils::Alarm *alarm =
    new sentinel::utils::Alarm(level, exception, _app);
  try
  {
    _alarmInfoSpace->fireItemAvailable(name, alarm);
  }
  catch(xdata::exception::Exception)
  {
    // Alarm is already set
    return;
  }
  
  #endif
}


void AlarmHandler::revokeAlarm
(
  const std::string name
)
{
  LOG4CPLUS_INFO(_app->getApplicationLogger(), "Revoking alarm " << name);

  #if SENTINELUTILS_VERSION_MAJOR>1
  
  sentinel::utils::Alarm *alarm;
  try
  {
    alarm = dynamic_cast<sentinel::utils::Alarm*>( _alarmInfoSpace->find( name ) );
  }
  catch(xdata::exception::Exception)
  {
    // Alarm has not been set
    return;
  }
  
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
