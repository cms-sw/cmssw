/**
 * This class manages the distribution of events to consumers from within
 * the storage manager.
 *
 * $Id: EventServer.cc,v 1.7 2007/11/09 23:09:12 badgett Exp $
 */

#include "EventFilter/StorageManager/interface/EventServer.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <iostream>
#include <boost/algorithm/string/case_conv.hpp>

using namespace std;
using namespace stor;
using namespace edm;

/**
 * Initialize the maximum accept interval.
 */
const double EventServer::MAX_ACCEPT_INTERVAL = 86400.0;  // seconds in 1 day

/**
 * EventServer constructor.  Throttling events are supported:
 * specifying a maximimum allowed rate of accepted events
 */
EventServer::EventServer(double maximumRate)
{
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
  disconnectedConsumerTestCounter_ = 0;

  selTableStringSize_ = 0;
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
  uint32 consumerId = consumer->getConsumerId();
  consumerTable_[consumerId] = consumer;
}

std::map< uint32, boost::shared_ptr<ConsumerPipe> > EventServer::getConsumerTable()
{
  return(consumerTable_);
}

/**
 * Returns a shared pointer to the consumer pipe with the specified ID
 * or an empty pointer if the ID was not found.
 */
boost::shared_ptr<ConsumerPipe> EventServer::getConsumer(uint32 consumerId)
{
  // initial empty pointer
  boost::shared_ptr<ConsumerPipe> consPtr;

  // lookup the consumer
  std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
  consIter = consumerTable_.find(consumerId);
  if (consIter != consumerTable_.end())
  {
    consPtr = consIter->second;
  }

  // return the pointer
  return consPtr;
}

/**
 * Processes the specified event.  This includes checking whether
 * the event is allowed to be delivered to consumers based on the
 * maximum event rate specified in the constructor, checking if
 * any consumers are ready to receive events, checking if any consumers
 * are interested in this specific event, making a local copy of the
 * event, and saving the event to be delivered to consumers.
 */
void EventServer::processEvent(const EventMsgView &eventView)
{
  // check if we are ready to accept another event
  struct timeval now;
  // throttle events that occur more frequently than our max allowed rate
  struct timezone dummyTZ;
  gettimeofday(&now, &dummyTZ);
  double timeDiff = (double) now.tv_sec;
  timeDiff -= (double) lastAcceptedEventTime_.tv_sec;
  timeDiff += ((double) now.tv_usec / 1000000.0);
  timeDiff -= ((double) lastAcceptedEventTime_.tv_usec / 1000000.0);
  if (timeDiff < minTimeBetweenEvents_) {return;}

  // do nothing if the event is empty
  if (eventView.size() == 0) {return;}

  // loop over the consumers in our list, and for each one check whether
  // it is ready for an event and if it wants this specific event.  If so,
  // create a local copy of the event (if not already done) and pass it
  // to the consumer pipe.
  boost::shared_ptr< vector<char> > bufPtr;
  std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
  for (consIter = consumerTable_.begin();
       consIter != consumerTable_.end();
       consIter++)
  {
    // test if the consumer is ready and wants the event
    boost::shared_ptr<ConsumerPipe> consPipe = consIter->second;
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
        lastAcceptedEventTime_ = now;
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

    // determine which consumers have disconnected
    std::vector<uint32> disconnectList;
    std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
    for (consIter = consumerTable_.begin();
         consIter != consumerTable_.end();
         consIter++)
    {
      boost::shared_ptr<ConsumerPipe> consPipe = consIter->second;
      FDEBUG(5) << "Checking if consumer " << consPipe->getConsumerId() <<
        " has disconnected " << std::endl;
      if (consPipe->isDisconnected())
      {
        disconnectList.push_back(consIter->first);
      }
    }

    // remove disconnected consumers from the consumer table
    std::vector<uint32>::const_iterator listIter;
    for (listIter = disconnectList.begin();
         listIter != disconnectList.end();
         listIter++)
    {
      uint32 consumerId = *listIter;
      consumerTable_.erase(consumerId);
    }
  }
}

/**
 * Returns the next event for the specified consumer.
 */
boost::shared_ptr< std::vector<char> > EventServer::getEvent(uint32 consumerId)
{
  // initial empty buffer
  boost::shared_ptr< vector<char> > bufPtr;

  // lookup the consumer
  std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
  consIter = consumerTable_.find(consumerId);
  if (consIter != consumerTable_.end())
  {
    boost::shared_ptr<ConsumerPipe> consPipe = consIter->second;
    bufPtr = consPipe->getEvent();
  }

  // return the event buffer
  return bufPtr;
}

void EventServer::clearQueue()
{
  std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator consIter;
  for (consIter = consumerTable_.begin();
       consIter != consumerTable_.end();
       consIter++)
  {
    boost::shared_ptr<ConsumerPipe> consPipe = consIter->second;
    consPipe->clearQueue();
  }
}

void EventServer::setStreamSelectionTable(std::map<std::string, Strings> const& selTable)
{
  streamSelectionTable_ = selTable;
  selTableStringSize_ = 0;
  std::map<std::string, Strings>::const_iterator mapIter;
  for (mapIter = selTable.begin(); mapIter != selTable.end(); mapIter++)
  {
    std::string streamLabel = mapIter->first;
    selTableStringSize_ += streamLabel.size();
    Strings selectionList = mapIter->second;
    for (unsigned int idx = 0; idx < selectionList.size(); idx++)
    {
      std::string selection = selectionList[idx];
      selTableStringSize_ += selection.size();
    }
  }
}

Strings EventServer::updateTriggerSelectionForStreams(Strings const& selectionList)
{
  Strings modifiedList;
  for (unsigned int idx = 0; idx < selectionList.size(); idx++) {
    std::string selection = selectionList[idx];
    std::string lcSelection = boost::algorithm::to_lower_copy(selection);
    if (lcSelection.find("stream", 0) == 0) {
      std::string streamLabel = selection.substr(6);
      std::map<std::string, Strings>::const_iterator mapIter =
        streamSelectionTable_.find(streamLabel);
      if (mapIter != streamSelectionTable_.end()) {
        Strings streamSelectionList = mapIter->second;
        for (unsigned int jdx = 0; jdx < streamSelectionList.size(); jdx++) {
          modifiedList.push_back(streamSelectionList.at(jdx));
        }
      }
      else {
        modifiedList.push_back(selection);
      }
    }
    else {
      modifiedList.push_back(selection);
    }
  }
  return modifiedList;
}
