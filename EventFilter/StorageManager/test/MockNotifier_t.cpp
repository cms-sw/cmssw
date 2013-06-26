#include <iostream>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/test/MockAlarmHandler.h"
#include "EventFilter/StorageManager/test/MockApplication.h"

using namespace std;
using namespace stor;

int main()
{

  xdaq::Application* app = mockapps::getMockXdaqApplication();

  boost::shared_ptr<StatisticsReporter> sr;
  SharedResourcesPtr sharedResources(new SharedResources());
  sharedResources->alarmHandler_.reset( new MockAlarmHandler() );
  sharedResources->configuration_.reset(
    new Configuration(app->getApplicationInfoSpace(), 0)
  );
  sr.reset( new StatisticsReporter( app, sharedResources ) );

  XCEPT_DECLARE( stor::exception::UnwantedEvents, xcept,
		 "Event is not tagged for any stream or consumer" );
  sharedResources->alarmHandler_->notifySentinel( AlarmHandler::ERROR, xcept );

  return 0;

}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
