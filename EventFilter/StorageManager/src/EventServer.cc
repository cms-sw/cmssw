/**
 * This class manages the distribution of events to consumers from within
 * the storage manager.
 *
 * 17-Aug-2006 - KAB  - Initial Implementation
 */

#include "EventFilter/StorageManager/interface/EventServer.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

using namespace std;
using namespace stor;
using namespace edm;

/**
 * Initialize the maximum reduction factor.
 */
const int EventServer::MAX_REDUCTION_FACTOR = 1000000;

/**
 * Initialize the maximum accept interval.
 */
const double EventServer::MAX_ACCEPT_INTERVAL = 86400.0;  // seconds in 1 day

/**
 * EventServer constructor.  Two ways of throttling events are supported:
 * specifying a maximimum allowed rate of accepted events and specifying
 * a fixed prescale.  If the fixed prescale value is greater than zero,
 * it takes precendence.  That is, the maximum rate is ignored if the
 * prescale is in effect.
 */
EventServer::EventServer(int eventPrescaleFactor, double maximumRate)
{
  // assign the prescale for candidate events
  // (zero and negative values are legal, and these signal that we should
  // use the maximum rate rather than the prescale)
  if (eventPrescaleFactor > MAX_REDUCTION_FACTOR)
  {
    eventReductionFactor_ = MAX_REDUCTION_FACTOR;
  }
  else
  {
    eventReductionFactor_ = eventPrescaleFactor;
  }

  // determine the amount of time that we need to wait between accepted
  // events (to ensure that the event server doesn't send "too many" events
  // to consumers).  The maximum rate specified to this constructor is
  // converted to an interval that is used internally, and the interval
  // is required to be somewhat reasonable.
  if (maximumRate < (1.0 / MAX_ACCEPT_INTERVAL))
  {
    minTimeBetweenEvents_ = MAX_ACCEPT_INTERVAL;
  }
  else
  {
    minTimeBetweenEvents_ = 1.0 / maximumRate;  // seconds
  }

  // initialize the last accepted event time to construction time
  struct timezone dummyTZ;
  gettimeofday(&lastAcceptedEventTime_, &dummyTZ);

  // initialize counters
  skippedEventCounter_ = 0;;
  disconnectedConsumerTestCounter_ = 0;
}

/**
 * EventServer destructor.
 */
EventServer::~EventServer()
{
  FDEBUG(5) << "Executing destructor for event server " << std::endl;
}

/**
 * Adds the specified consumer to the event server.
 */
void EventServer::addConsumer(boost::shared_ptr<ConsumerPipe> consumer)
{
  consumerList.push_back(consumer);
}

/**
 * Processes the specified event.  This includes checking whether
 * the event is allowed to be delivered to consumers based on the
 * prescale and 
 * maximum event rate specified in the constructor, checking if
 * any consumers are ready to receive events, checking if any consumers
 * are interested in this specific event, making a local copy of the
 * event, and saving the event to be delivered to consumers.
 */
void EventServer::processEvent(const EventMsgView &eventView)
{
  // check if we are ready to accept another event
  struct timeval now;
  if (eventReductionFactor_ > 0)
  {
    // skip events based on the specified reduction factor
    skippedEventCounter_++;
    if (skippedEventCounter_ >= eventReductionFactor_)
    {
      // "accept" this event and reset the counter
      skippedEventCounter_ = 0;
    }
    else
    {
      // skip this event
      return;
    }
  }
  else
  {
    // throttle events that occur more frequently than our max allowed rate
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    double timeDiff = (double) now.tv_sec;
    timeDiff -= (double) lastAcceptedEventTime_.tv_sec;
    timeDiff += ((double) now.tv_usec / 1000000.0);
    timeDiff -= ((double) lastAcceptedEventTime_.tv_usec / 1000000.0);
    //cout << "timeDiff = " << timeDiff <<
    //  ", minTime = " << minTimeBetweenEvents_ << std::endl;
    if (timeDiff < minTimeBetweenEvents_) {return;}
  }

  // do nothing if the event is empty
  if (eventView.size() == 0) {return;}

  // loop over the consumers in our list, and for each one check whether
  // it is ready for an event and if it wants this specific event.  If so,
  // create a local copy of the event (if not already done) and pass it
  // to the consumer pipe.
  boost::shared_ptr< vector<char> > bufPtr;
  std::vector< boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
  for (consIter = consumerList.begin();
       consIter != consumerList.end();
       consIter++)
  {
    // test if the consumer is ready and wants the event
    boost::shared_ptr<ConsumerPipe> consPipe = *consIter;
    FDEBUG(5) << "Checking if consumer " << consPipe->getConsumerId() <<
      " wants event " << eventView.event() << std::endl;
    if (consPipe->isReadyForEvent() &&
        consPipe->wantsEvent(eventView))
    {
      // check if we need to make a local copy of the event
      if (bufPtr.get() == NULL)
      {
        FDEBUG(5) << "Creating a buffer for event " <<
          eventView.event() << std::endl;

        // create a local buffer of the appropriate size
        boost::shared_ptr< vector<char> >
          tmpBufPtr(new vector<char>(eventView.size()));

        // copy the data to the local buffer
        unsigned char *target = (unsigned char *) &(*tmpBufPtr)[0];
        unsigned char *source = eventView.startAddress();
        int dataSize = eventView.size();
        std::copy(source, source+dataSize, target);

        // switch buffers
        bufPtr.swap(tmpBufPtr);

        // update the local time stamp for the latest accepted event
        if (eventReductionFactor_ <= 0)
        {
          lastAcceptedEventTime_ = now;
        }
      }

      // add the event to the consumer pipe
      consPipe->putEvent(bufPtr);
    }
  }

  // periodically check for disconnected consumers
  disconnectedConsumerTestCounter_++;
  if (disconnectedConsumerTestCounter_ >= 500)
  {
    // reset counter
    disconnectedConsumerTestCounter_ = 0;

#if 0
// 24-Aug-2006, KAB - need to understand std::vector.erase better so we
// can use it instead of the vector copy below

    // loop over the list of consumers in reverse order so that when
    // we remove an element from the list, it only affects the position
    // of elements that we've already processed
    std::vector< boost::shared_ptr<ConsumerPipe> >::reverse_iterator revIter;
    for (revIter = consumerList.rbegin();
         revIter != consumerList.rend();
         revIter++)
    {
      // test if the consumer has disconnected
      boost::shared_ptr<ConsumerPipe> consPipe = *revIter;
      FDEBUG(5) << "Checking if consumer " << consPipe->getConsumerId() <<
        " has disconnected " << std::endl;
      if (consPipe->isDisconnected())
      {
        consumerList.erase(revIter);
      }
    }
#endif

    std::vector< boost::shared_ptr<ConsumerPipe> > activeList;
    std::vector< boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
    for (consIter = consumerList.begin();
         consIter != consumerList.end();
         consIter++)
    {
      boost::shared_ptr<ConsumerPipe> consPipe = *consIter;
      FDEBUG(5) << "Checking if consumer " << consPipe->getConsumerId() <<
        " has disconnected " << std::endl;
      if (! (consPipe->isDisconnected()))
      {
        activeList.push_back(consPipe);
      }
    }
    consumerList = activeList;
  }
}

/**
 * Returns the next event for the specified consumer.
 */
boost::shared_ptr< std::vector<char> >
  EventServer::getEvent(uint32 consumerId)
{
  // TODO - convert to a hashtable so that we can look up the consumer
  // by its ID
  boost::shared_ptr< vector<char> > bufPtr;
  std::vector< boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
  for (consIter = consumerList.begin();
       consIter != consumerList.end();
       consIter++)
  {
    boost::shared_ptr<ConsumerPipe> consPipe = *consIter;
    if (consPipe->getConsumerId() == consumerId)
    {
      bufPtr = consPipe->getEvent();
      break;
    }
  }
  return bufPtr;
}
