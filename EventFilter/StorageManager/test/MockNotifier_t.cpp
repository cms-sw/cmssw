#include <iostream>

#include "EventFilter/StorageManager/interface/StatisticsReporter.h"              
#include "EventFilter/StorageManager/test/MockApplication.h"
#include "EventFilter/StorageManager/interface/Exception.h"

using namespace std;
using namespace stor;

int main()
{

  xdaq::Application* app = mockapps::getMockXdaqApplication();

  boost::shared_ptr<StatisticsReporter> sr;
  sr.reset( new StatisticsReporter( app, 0 ) );

  XCEPT_DECLARE( stor::exception::UnwantedEvents, xcept,
		 "Event is not tagged for any stream or consumer" );
  sr->alarmHandler()->notifySentinel( AlarmHandler::ERROR, xcept );

  return 0;

}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
