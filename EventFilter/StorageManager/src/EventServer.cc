/**
 * This class manages the distribution of events to consumers from within
 * the storage manager.
 *
 * 17-Aug-2006 - KAB  - Initial Implementation
 */

#include "EventFilter/StorageManager/interface/EventServer.h"

using namespace std;
using namespace stor;

/**
 * EventServer constructor.
 */
EventServer::EventServer(double maximumRate, int vipConsumerQueueSize,
                         double consumerIdleTime, double consumerDisconnectTime):
  maximumTotalRate_(maximumRate), vipQueueSize_(vipConsumerQueueSize),
  consumerIdleTime_(consumerIdleTime), consumerDisconnectTime_(consumerDisconnectTime)
{
}

/**
 * EventServer destructor.
 */
EventServer::~EventServer()
{
}

/**
 * Adds the specified consumer to the event server.
 */
void EventServer::addConsumer(boost::shared_ptr<ConsumerPipe> consumer)
{
  consumerList.push_back(consumer);
}

/**
 * Tests if any registered consumer wants the specified event.
 */
bool EventServer::wantsEvent(const EventMsgView &eventView)
{
  std::vector< boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
  for (consIter = consumerList.begin(); consIter != consumerList.end(); consIter++) {
    boost::shared_ptr<ConsumerPipe> pipePtr = *consIter;
    cout << "Checking if consumer " << pipePtr->getConsumerId() <<
      " wants event " << eventView.event() << endl;
  }

  return false;
}
