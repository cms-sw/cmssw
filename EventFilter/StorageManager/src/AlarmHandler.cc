//$Id: AlarmHandler.cc,v 1.8 2009/09/29 08:04:58 mommsen Exp $
/// @file: AlarmHandler.cc


#include "sentinel/utils/version.h"
#if SENTINELUTILS_VERSION_MAJOR>1
#include "sentinel/utils/Alarm.h"
#endif

#include "xcept/tools.h"
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


void AlarmHandler::notifySentinel
(
  const ALARM_LEVEL level,
  xcept::Exception& exception
)
{

  switch( level )
  {
    case OKAY:
      LOG4CPLUS_INFO(_app->getApplicationLogger(),
        xcept::stdformat_exception_history(exception));
      break;

    case WARNING:
      LOG4CPLUS_WARN(_app->getApplicationLogger(),
        xcept::stdformat_exception_history(exception));
      _app->notifyQualified("warning", exception);
      break;

    case ERROR:
      LOG4CPLUS_ERROR(_app->getApplicationLogger(),
        xcept::stdformat_exception_history(exception));
      _app->notifyQualified("error", exception);
      break;

    case FATAL:
      LOG4CPLUS_FATAL(_app->getApplicationLogger(),
        xcept::stdformat_exception_history(exception));
      _app->notifyQualified("fatal", exception);
      break;

    default:
      LOG4CPLUS_WARN(_app->getApplicationLogger(),
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

  #else

  return false;

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


void AlarmHandler::clearAllAlarms()
{
  if (!_alarmInfoSpace) return;

  boost::mutex::scoped_lock sl( _mutex );

  typedef std::map<std::string, xdata::Serializable*, std::less<std::string> > alarmList;
  alarmList alarms = _alarmInfoSpace->match(".*");
  for (alarmList::const_iterator it = alarms.begin(), itEnd = alarms.end();
       it != itEnd; ++it)
  {
    sentinel::utils::Alarm* alarm = dynamic_cast<sentinel::utils::Alarm*>(it->second);
    _alarmInfoSpace->fireItemRevoked(it->first, _app);
    delete alarm;
  }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
