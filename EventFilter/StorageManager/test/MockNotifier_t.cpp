#include <iostream>

#include "EventFilter/StorageManager/interface/StatisticsReporter.h"              
#include "EventFilter/StorageManager/test/MockApplication.h"
#include "EventFilter/StorageManager/interface/Exception.h"

using namespace std;
using namespace stor;

int main()
{

  MockApplicationStub* stub( new MockApplicationStub() );
  MockApplication* app( new MockApplication( stub ) );

  boost::shared_ptr<StatisticsReporter> sr;
  sr.reset( new StatisticsReporter( app, 0 ) );

  XCEPT_DECLARE( stor::exception::UnwantedEvent, xcept,
		 "Event is not tagged for any stream or consumer" );
  sr->alarmHandler()->notifySentinel( AlarmHandler::ERROR, xcept );

  return 0;

}
