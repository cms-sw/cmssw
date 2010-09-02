/**
 * This class manages the distribution of DQM events to consumers from within
 * the storage manager or the SM Proxy Server.
 *
 * Initial Implementation based on Kurt's EventServer
 * make a common class later when all this works
 *
 * $Id: DQMEventServer.cc,v 1.6 2010/05/17 15:59:10 mommsen Exp $
 *
 * Note: this class is no longer used in the StorageManager, but is still
 * required by the SMProxyServer (Remi Mommsen, May 5, 2009)
 */

#include "EventFilter/StorageManager/interface/DQMEventServer.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <iostream>

using namespace std;
using namespace stor;
using namespace edm;

/**
 * Initialize the maximum accept interval.
 */
const double DQMEventServer::MAX_ACCEPT_INTERVAL = 86400.0;  // seconds in 1 day

/**
 * DQMEventServer constructor.  One way of throttling DQMevents is supported:
 * specifying a maximimum allowed rate of accepted event. However it is not
 * yet implemented because it should be a maximum rate of accepted updates,
 * since one update can have multiple DQMEvents with the same folderID but
 * different eventAtUpdate from different FUs are about the same time and we
 * want to accept all these to collate them
 * Needs implementation to handle if already collated.
 * Also means we need a buffer that can hold more than one DQMEvent
 */
DQMEventServer::DQMEventServer(double maximumRate)
{
  // determine the amount of time that we need to wait between accepted
  // events (to ensure that the event server doesn't send "too many" events
  // to consumers).  The maximum rate specified to this constructor is
  // converted to an interval that is used internally, and the interval
  // is required to be somewhat reasonable.
  /* Note that this is not
   * yet implemented because it should be a maximum rate of accepted updates,
   * since one update can have multiple DQMEvents with the same folderID but
   * different eventAtUpdate from different FUs are about the same time and we
   * want to accept all these to collate them
   * Needs implementation to handle if already collated.
   */
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
}

/**
 * DQMEventServer destructor.
 */
DQMEventServer::~DQMEventServer()
{
  FDEBUG(5) << "Executing destructor for DQMevent server " << std::endl;
}

/**
 * Adds the specified consumer to the event server.
 */
void DQMEventServer::addConsumer(boost::shared_ptr<DQMConsumerPipe> consumer)
{
  uint32_t consumerId = consumer->getConsumerId();
  consumerTable[consumerId] = consumer;
}

/**
 * Returns a shared pointer to the consumer pipe with the specified ID
 * or an empty pointer if the ID was not found.
 */
boost::shared_ptr<DQMConsumerPipe> DQMEventServer::getConsumer(uint32_t consumerId)
{
  // initial empty pointer
  boost::shared_ptr<DQMConsumerPipe> consPtr;

  // lookup the consumer
  std::map< uint32_t, boost::shared_ptr<DQMConsumerPipe> >::const_iterator consIter;
  consIter = consumerTable.find(consumerId);
  if (consIter != consumerTable.end())
  {
    consPtr = consIter->second;
  }

  // return the pointer
  return consPtr;
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
void DQMEventServer::processDQMEvent(const DQMEventMsgView &eventView)
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
  //std::cout << "timeDiff = " << timeDiff <<
  //  ", minTime = " << minTimeBetweenEvents_ << std::endl;
  //if (timeDiff < minTimeBetweenEvents_) {return;}

  // actually do not throttle for now as DQMEvents all come at once
  // for different top level folders and we can miss the second and
  // subsequent top level folders for the same update if we throttle this
  // way, we should include the updateAtEventNumber in deciding
  // we need a buffer that can hold more than one DQMEvent because all
  // FUs are going to send their same folderID DQMEvents at once

  // do nothing if the event is empty
  if (eventView.size() == 0) {return;}

  // loop over the consumers in our list, and for each one check whether
  // it is ready for an event and if it wants this specific event.  If so,
  // create a local copy of the event (if not already done) and pass it
  // to the consumer pipe.
  boost::shared_ptr< std::vector<char> > bufPtr;
  std::map< uint32_t, boost::shared_ptr<DQMConsumerPipe> >::const_iterator consIter;
  for (consIter = consumerTable.begin();
       consIter != consumerTable.end();
       consIter++)
  {
    // test if the consumer is ready and wants the event
    boost::shared_ptr<DQMConsumerPipe> consPipe = consIter->second;
    FDEBUG(5) << "Checking if consumer " << consPipe->getConsumerId() <<
      " wants update " << eventView.eventNumberAtUpdate() << 
      " and folder " << eventView.topFolderName() << std::endl;
    
    if (consPipe->isReadyForEvent() &&
        consPipe->wantsDQMEvent(eventView))
    {
      // check if we need to make a local copy of the event
      if (bufPtr.get() == NULL)
      {
        FDEBUG(5) << "Creating a buffer for update " <<
          eventView.eventNumberAtUpdate() << 
          " and folder " << eventView.topFolderName() <<std::endl;

        // create a local buffer of the appropriate size
        boost::shared_ptr< std::vector<char> >
          tmpBufPtr(new std::vector<char>(eventView.size()));

        // copy the data to the local buffer
        unsigned char *target = (unsigned char *) &(*tmpBufPtr)[0];
        unsigned char *source = eventView.startAddress();
        int dataSize = eventView.size();
        std::copy(source, source+dataSize, target);

        // switch buffers
        bufPtr.swap(tmpBufPtr); // what happens to the memory and do we need a delete?

        // update the local time stamp for the latest accepted event
        lastAcceptedEventTime_ = now;
      }
      FDEBUG(5) << "Adding DQMevent to consumer pipe for update " <<
          eventView.eventNumberAtUpdate() << 
          " and folder " << eventView.topFolderName() <<std::endl;

      // add the event to the consumer pipe
      consPipe->putDQMEvent(bufPtr);
    }
  }

  // periodically check for disconnected consumers
  disconnectedConsumerTestCounter_++;
  if (disconnectedConsumerTestCounter_ >= 500)
  {
    // reset counter
    disconnectedConsumerTestCounter_ = 0;

    // determine which consumers have disconnected
    std::vector<uint32_t> disconnectList;
    std::map< uint32_t, boost::shared_ptr<DQMConsumerPipe> >::const_iterator consIter;
    for (consIter = consumerTable.begin();
         consIter != consumerTable.end();
         consIter++)
    {
      boost::shared_ptr<DQMConsumerPipe> consPipe = consIter->second;
      FDEBUG(5) << "Checking if DQM consumer " << consPipe->getConsumerId() <<
        " has disconnected " << std::endl;
      if (consPipe->isDisconnected())
      {
        disconnectList.push_back(consIter->first);
      }
    }

    // remove disconnected consumers from the consumer table
    std::vector<uint32_t>::const_iterator listIter;
    for (listIter = disconnectList.begin();
         listIter != disconnectList.end();
         listIter++)
    {
      uint32_t consumerId = *listIter;
      consumerTable.erase(consumerId);
    }
  }
}

/**
 * Returns the next event for the specified consumer.
 */
boost::shared_ptr< std::vector<char> > DQMEventServer::getDQMEvent(uint32_t consumerId)
{
  // initial empty buffer
  boost::shared_ptr< std::vector<char> > bufPtr;

  // lookup the consumer
  std::map< uint32_t, boost::shared_ptr<DQMConsumerPipe> >::const_iterator consIter;
  consIter = consumerTable.find(consumerId);
  if (consIter != consumerTable.end())
  {
    boost::shared_ptr<DQMConsumerPipe> consPipe = consIter->second;
    bufPtr = consPipe->getDQMEvent();
  }

  // return the event buffer
  return bufPtr;
}

void DQMEventServer::clearQueue()
{
  std::map< uint32_t, boost::shared_ptr<DQMConsumerPipe> >::const_iterator consIter;
  for (consIter = consumerTable.begin();
       consIter != consumerTable.end();
       consIter++)
  {
    boost::shared_ptr<DQMConsumerPipe> consPipe = consIter->second;
    consPipe->clearQueue();
  }
}
