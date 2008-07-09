/**
 * This class manages the distribution of events to consumers from within
 * the storage manager.
 *
 * $Id: EventServer.cc,v 1.9 2008/04/16 16:14:08 biery Exp $
 */

#include "EventFilter/StorageManager/interface/EventServer.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <iostream>
#include <boost/algorithm/string/case_conv.hpp>
#include "zlib.h"

using namespace std;
using namespace stor;
using namespace edm;

/**
 * EventServer constructor.  Throttling events are supported:
 * specifying a maximimum allowed rate of accepted events
 */
EventServer::EventServer(double maxEventRate, double maxDataRate,
                         std::string hltOutputSelection)
{
  // initialize counters
  disconnectedConsumerTestCounter_ = 0;

  selTableStringSize_ = 0;

  longTermInputCounter_.reset(new ForeverCounter());
  shortTermInputCounter_.reset(new RollingIntervalCounter(180,5,20));
  longTermOutputCounter_.reset(new ForeverCounter());
  shortTermOutputCounter_.reset(new RollingIntervalCounter(180,5,20));

  //rateLimiter_.reset(new RateLimiter(maxEventRate, maxDataRate));
  this->maxEventRate_ = maxEventRate;
  this->maxDataRate_ = maxDataRate;
  this->hltOutputSelection_ = hltOutputSelection;

  uLong crc = crc32(0L, Z_NULL, 0);
  Bytef* crcbuf = (Bytef*) hltOutputSelection.data();
  crc = crc32(crc, crcbuf, hltOutputSelection.length());
  this->hltOutputModuleId_ = static_cast<uint32>(crc);

  outsideTimer_.reset();
  insideTimer_.reset();

  // initialize the counters that we use for statistics
  longTermInsideCPUTimeCounter_.reset(new ForeverCounter());
  shortTermInsideCPUTimeCounter_.reset(new RollingIntervalCounter(180,5,20));
  longTermInsideRealTimeCounter_.reset(new ForeverCounter());
  shortTermInsideRealTimeCounter_.reset(new RollingIntervalCounter(180,5,20));
  longTermOutsideCPUTimeCounter_.reset(new ForeverCounter());
  shortTermOutsideCPUTimeCounter_.reset(new RollingIntervalCounter(180,5,20));
  longTermOutsideRealTimeCounter_.reset(new ForeverCounter());
  shortTermOutsideRealTimeCounter_.reset(new RollingIntervalCounter(180,5,20));

  generator_.reset(new boost::uniform_01<boost::mt19937>(baseGenerator_));
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

  // add the consumer (by ID) to the rateLimiter instance that we use
  // to provide a fair share of the limited bandwidth to each consumer.
  //rateLimiter_->addConsumer(consumerId);
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
  // do nothing if the event is empty
  if (eventView.size() == 0) {return;}

  // the event must be from the correct HLT output module
  if (eventView.outModId() != hltOutputModuleId_) {return;}

  // stop the timer that we use to measure CPU and real time outside the ES
  outsideTimer_.stop();

  // stop the timer that we use to measure CPU and real time inside the ES
  insideTimer_.start();

  // add the event to our statistics for events that are input to the ES
  double sizeInMB = static_cast<double>(eventView.size()) / 1048576.0;
  double now = BaseCounter::getCurrentTime();
  longTermInputCounter_->addSample(sizeInMB);
  shortTermInputCounter_->addSample(sizeInMB, now);

  // prescale events based on the input event and data rates
  double eventRate = shortTermInputCounter_->getSampleRate(now);
  double dataRate = shortTermInputCounter_->getValueRate(now);
  double eventRatePrescale = eventRate / maxEventRate_;
  double dataRatePrescale = dataRate / maxDataRate_;
  double effectivePrescale = std::max(eventRatePrescale, dataRatePrescale);
  if (effectivePrescale > 1.0) {
    double instantRatio = 1.0 / effectivePrescale;
    double randValue = (*generator_)();
    if (randValue > instantRatio) {
      // track timer statistics and start/stop timers as appropriate
      insideTimer_.stop();
      longTermInsideCPUTimeCounter_->addSample(insideTimer_.cpuTime());
      shortTermInsideCPUTimeCounter_->addSample(insideTimer_.cpuTime(), now);
      longTermInsideRealTimeCounter_->addSample(insideTimer_.realTime());
      shortTermInsideRealTimeCounter_->addSample(insideTimer_.realTime(), now);
      longTermOutsideCPUTimeCounter_->addSample(outsideTimer_.cpuTime());
      shortTermOutsideCPUTimeCounter_->addSample(outsideTimer_.cpuTime(), now);
      longTermOutsideRealTimeCounter_->addSample(outsideTimer_.realTime());
      shortTermOutsideRealTimeCounter_->addSample(outsideTimer_.realTime(), now);
      outsideTimer_.reset();
      insideTimer_.reset();
      outsideTimer_.start();
      return;
    }
  }

  // loop over the consumers in our list, and for each one check whether
  // it is ready for an event and if it wants this specific event.  If so,
  // create a local copy of the event (if not already done) and pass it
  // to the consumer pipe.

  // determine which consumers are interested in the event
  std::vector<uint32> candidateList;
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
    if (consPipe->wantsEvent(eventView) &&
        consPipe->isReadyForEvent(now))
    {
      candidateList.push_back(consPipe->getConsumerId());
    }
  }

  // determine which of the candidate consumers are allowed
  // to receive another event at this time
  //std::vector<uint32> allowedList =
  //  rateLimiter_->getAllowedConsumersFromList(sizeInMB, candidateList);
  std::vector<uint32> allowedList = candidateList;

  // send the event to the allowed consumers
  for (uint32 idx = 0; idx < allowedList.size(); ++idx)
  {
    uint32 consumerId = allowedList[idx];

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
    }

    // add the event to the consumer pipe
    boost::shared_ptr<ConsumerPipe> consPipe = getConsumer(consumerId);
    consPipe->wasConsidered(now);
    consPipe->putEvent(bufPtr);

    // add the event to our statistics for "output" events
    // Adding the stats once per consumer is (currently) believed
    // to give a more accurate picture of what is being sent out.
    // (Even though we only have one copy of the event internally,
    // it uses up bandwidth N times for N consumers.)
    longTermOutputCounter_->addSample(sizeInMB);
    shortTermOutputCounter_->addSample(sizeInMB, now);
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

      // remove the consumer from the rateLimiter instance so that it is
      // no longer considered for a fair share of the allowed bandwidth
      //rateLimiter_->removeConsumer(consumerId);
    }
  }

  // track timer statistics and start/stop timers as appropriate
  now = BaseCounter::getCurrentTime();
  insideTimer_.stop();
  longTermInsideCPUTimeCounter_->addSample(insideTimer_.cpuTime());
  shortTermInsideCPUTimeCounter_->addSample(insideTimer_.cpuTime(), now);
  longTermInsideRealTimeCounter_->addSample(insideTimer_.realTime());
  shortTermInsideRealTimeCounter_->addSample(insideTimer_.realTime(), now);
  longTermOutsideCPUTimeCounter_->addSample(outsideTimer_.cpuTime());
  shortTermOutsideCPUTimeCounter_->addSample(outsideTimer_.cpuTime(), now);
  longTermOutsideRealTimeCounter_->addSample(outsideTimer_.realTime());
  shortTermOutsideRealTimeCounter_->addSample(outsideTimer_.realTime(), now);
  outsideTimer_.reset();
  insideTimer_.reset();
  outsideTimer_.start();
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

/**
 * Returns the number of events for the specified statistics types
 * (short term vs. long term; input vs. output).
 */
long long EventServer::getEventCount(STATS_TIME_FRAME timeFrame,
                                     STATS_SAMPLE_TYPE sampleType,
                                     double currentTime)
{
  if (timeFrame == SHORT_TERM_STATS) {
    if (sampleType == INPUT_STATS) {
      return shortTermInputCounter_->getSampleCount(currentTime);
    }
    else {
      return shortTermOutputCounter_->getSampleCount(currentTime);
    }
  }
  else {
    if (sampleType == INPUT_STATS) {
      return longTermInputCounter_->getSampleCount();
    }
    else {
      return longTermOutputCounter_->getSampleCount();
    }
  }
}

/**
 * Returns the rate of events for the specified statistics types
 * (short term vs. long term; input vs. output).
 */
double EventServer::getEventRate(STATS_TIME_FRAME timeFrame,
                                 STATS_SAMPLE_TYPE sampleType,
                                 double currentTime)
{
  if (timeFrame == SHORT_TERM_STATS) {
    if (sampleType == INPUT_STATS) {
      return shortTermInputCounter_->getSampleRate(currentTime);
    }
    else {
      return shortTermOutputCounter_->getSampleRate(currentTime);
    }
  }
  else {
    if (sampleType == INPUT_STATS) {
      return longTermInputCounter_->getSampleRate(currentTime);
    }
    else {
      return longTermOutputCounter_->getSampleRate(currentTime);
    }
  }
}

/**
 * Returns the data rate for the specified statistics types
 * (short term vs. long term; input vs. output).
 */
double EventServer::getDataRate(STATS_TIME_FRAME timeFrame,
                                STATS_SAMPLE_TYPE sampleType,
                                double currentTime)
{
  if (timeFrame == SHORT_TERM_STATS) {
    if (sampleType == INPUT_STATS) {
      return shortTermInputCounter_->getValueRate(currentTime);
    }
    else {
      return shortTermOutputCounter_->getValueRate(currentTime);
    }
  }
  else {
    if (sampleType == INPUT_STATS) {
      return longTermInputCounter_->getValueRate(currentTime);
    }
    else {
      return longTermOutputCounter_->getValueRate(currentTime);
    }
  }
}

/**
 * Returns the duration (in seconds) for the specified statistics types
 * (short term vs. long term; input vs. output).
 * "Duration" here means the length of time in which the specified
 * statistics have been collected.
 */
double EventServer::getDuration(STATS_TIME_FRAME timeFrame,
                                STATS_SAMPLE_TYPE sampleType,
                                double currentTime)
{
  if (timeFrame == SHORT_TERM_STATS) {
    if (sampleType == INPUT_STATS) {
      return shortTermInputCounter_->getDuration(currentTime);
    }
    else {
      return shortTermOutputCounter_->getDuration(currentTime);
    }
  }
  else {
    if (sampleType == INPUT_STATS) {
      return longTermInputCounter_->getDuration(currentTime);
    }
    else {
      return longTermOutputCounter_->getDuration(currentTime);
    }
  }
}

/**
 * Returns the amount of time spent processing events inside the
 * event server for the specified time frame (short or long term)
 * and timing type (CPU time or real time).
 */
double EventServer::getInternalTime(STATS_TIME_FRAME timeFrame,
                                    STATS_TIMING_TYPE timingType,
                                    double currentTime)
{
  if (timeFrame == SHORT_TERM_STATS) {
    if (timingType == CPUTIME) {
      return shortTermInsideCPUTimeCounter_->getValueSum(currentTime);
    }
    else {
      return shortTermInsideRealTimeCounter_->getValueSum(currentTime);
    }
  }
  else {
    if (timingType == CPUTIME) {
      return longTermInsideCPUTimeCounter_->getValueSum();
    }
    else {
      return longTermInsideRealTimeCounter_->getValueSum();
    }
  }
}

/**
 * Returns the total time used by the current process (both inside
 * and outside the event server) for the specified time frame
 * (short or long term) and timing type (CPU time or real time).
 */
double EventServer::getTotalTime(STATS_TIME_FRAME timeFrame,
                                 STATS_TIMING_TYPE timingType,
                                 double currentTime)
{
  if (timeFrame == SHORT_TERM_STATS) {
    if (timingType == CPUTIME) {
      double insideTime =
        shortTermInsideCPUTimeCounter_->getValueSum(currentTime);
      double outsideTime =
        shortTermOutsideCPUTimeCounter_->getValueSum(currentTime);
      return (insideTime + outsideTime);
    }
    else {
      double insideTime =
        shortTermInsideRealTimeCounter_->getValueSum(currentTime);
      double outsideTime =
        shortTermOutsideRealTimeCounter_->getValueSum(currentTime);
      return (insideTime + outsideTime);
    }
  }
  else {
    if (timingType == CPUTIME) {
      double insideTime =
        longTermInsideCPUTimeCounter_->getValueSum();
      double outsideTime =
        longTermOutsideCPUTimeCounter_->getValueSum();
      return (insideTime + outsideTime);
    }
    else {
      double insideTime =
        longTermInsideRealTimeCounter_->getValueSum();
      double outsideTime =
        longTermOutsideRealTimeCounter_->getValueSum();
      return (insideTime + outsideTime);
    }
  }
}

/**
 * Returns the fraction of time spent processing events inside the
 * event server for the specified time frame (short or long term)
 * and timing type (CPU time or real time).
 */
double EventServer::getTimeFraction(STATS_TIME_FRAME timeFrame,
                                    STATS_TIMING_TYPE timingType,
                                    double currentTime)
{
  if (timeFrame == SHORT_TERM_STATS) {
    if (timingType == CPUTIME) {
      double insideTime =
        shortTermInsideCPUTimeCounter_->getValueSum(currentTime);
      double outsideTime =
        shortTermOutsideCPUTimeCounter_->getValueSum(currentTime);
      if (outsideTime > 0.0) {
        return (insideTime / (insideTime + outsideTime));
      }
      else {
        return 0.0;
      }
    }
    else {
      double insideTime =
        shortTermInsideRealTimeCounter_->getValueSum(currentTime);
      double outsideTime =
        shortTermOutsideRealTimeCounter_->getValueSum(currentTime);
      if (outsideTime > 0.0) {
        return (insideTime / (insideTime + outsideTime));
      }
      else {
        return 0.0;
      }
    }
  }
  else {
    if (timingType == CPUTIME) {
      double insideTime =
        longTermInsideCPUTimeCounter_->getValueSum();
      double outsideTime =
        longTermOutsideCPUTimeCounter_->getValueSum();
      if (outsideTime > 0.0) {
        return (insideTime / (insideTime + outsideTime));
      }
      else {
        return 0.0;
      }
    }
    else {
      double insideTime =
        longTermInsideRealTimeCounter_->getValueSum();
      double outsideTime =
        longTermOutsideRealTimeCounter_->getValueSum();
      if (outsideTime > 0.0) {
        return (insideTime / (insideTime + outsideTime));
      }
      else {
        return 0.0;
      }
    }
  }
}
