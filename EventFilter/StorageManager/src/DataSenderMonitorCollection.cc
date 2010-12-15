// $Id: DataSenderMonitorCollection.cc,v 1.15 2010/12/10 14:31:52 mommsen Exp $
/// @file: DataSenderMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include <zlib.h>
#include <boost/lexical_cast.hpp>

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"

using namespace stor;


DataSenderMonitorCollection::DataSenderMonitorCollection
(
  const utils::duration_t& updateInterval,
  boost::shared_ptr<AlarmHandler> ah
) :
MonitorCollection(updateInterval),
_connectedRBs(0),
_connectedEPs(0),
_activeEPs(0),
_outstandingDataDiscards(0),
_outstandingDQMDiscards(0),
_faultyEvents(0),
_ignoredDiscards(0),
_updateInterval(updateInterval),
_alarmHandler(ah)
{}


void DataSenderMonitorCollection::addFragmentSample(I2OChain const& i2oChain)
{
  // focus only on INIT and event fragments, for now
  if (i2oChain.messageCode() != Header::INIT &&
      i2oChain.messageCode() != Header::EVENT) {return;}
  if (i2oChain.fragmentCount() != 1) {return;}

  // fetch basic data from the I2OChain
  //double fragmentSize = static_cast<double>(i2oChain.totalDataSize());

  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  OutModRecordPtr topLevelOutModPtr, rbSpecificOutModPtr, fuSpecificOutModPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getAllNeededPointers(i2oChain, rbRecordPtr, fuRecordPtr,
                                            topLevelOutModPtr, rbSpecificOutModPtr,
                                            fuSpecificOutModPtr);
  }

  // accumulate the data of interest
  //if (pointersAreValid)
  //{
  //topLevelOutModPtr->fragmentSize.addSample(fragmentSize);
  //rbSpecificOutModPtr->fragmentSize.addSample(fragmentSize);
  //fuSpecificOutModPtr->fragmentSize.addSample(fragmentSize);
  //}
}


void DataSenderMonitorCollection::addInitSample(I2OChain const& i2oChain)
{
  // sanity checks
  if (i2oChain.messageCode() != Header::INIT) {return;}
  if (! i2oChain.complete()) {return;}

  // fetch basic data from the I2OChain
  std::string outModName = i2oChain.outputModuleLabel();
  uint32_t msgSize = i2oChain.totalDataSize();

  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  OutModRecordPtr topLevelOutModPtr, rbSpecificOutModPtr, fuSpecificOutModPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getAllNeededPointers(i2oChain, rbRecordPtr, fuRecordPtr,
                                            topLevelOutModPtr, rbSpecificOutModPtr,
                                            fuSpecificOutModPtr);
  }

  // accumulate the data of interest
  if (pointersAreValid)
  {
    topLevelOutModPtr->name = outModName;
    topLevelOutModPtr->initMsgSize = msgSize;

    ++rbRecordPtr->initMsgCount;
    rbSpecificOutModPtr->name = outModName;
    rbSpecificOutModPtr->initMsgSize = msgSize;

    ++fuRecordPtr->initMsgCount;
    fuSpecificOutModPtr->name = outModName;
    fuSpecificOutModPtr->initMsgSize = msgSize;
  }
}


void DataSenderMonitorCollection::addEventSample(I2OChain const& i2oChain)
{
  // sanity checks
  if (i2oChain.messageCode() != Header::EVENT) {return;}
  if (! i2oChain.complete()) {return;}

  // fetch basic data from the I2OChain
  double eventSize = static_cast<double>(i2oChain.totalDataSize());
  uint32_t runNumber = i2oChain.runNumber();
  uint32_t eventNumber = i2oChain.eventNumber();

  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  OutModRecordPtr topLevelOutModPtr, rbSpecificOutModPtr, fuSpecificOutModPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getAllNeededPointers(i2oChain, rbRecordPtr, fuRecordPtr,
                                            topLevelOutModPtr, rbSpecificOutModPtr,
                                            fuSpecificOutModPtr);
  }

  // accumulate the data of interest
  if (pointersAreValid)
  {
    topLevelOutModPtr->eventSize.addSample(eventSize);

    rbRecordPtr->lastRunNumber = runNumber;
    rbRecordPtr->lastEventNumber = eventNumber;
    rbRecordPtr->eventSize.addSample(eventSize);
    rbSpecificOutModPtr->eventSize.addSample(eventSize);

    fuRecordPtr->lastRunNumber = runNumber;
    fuRecordPtr->lastEventNumber = eventNumber;
    fuRecordPtr->shortIntervalEventSize.addSample(eventSize);
    fuRecordPtr->mediumIntervalEventSize.addSample(eventSize);
    fuSpecificOutModPtr->eventSize.addSample(eventSize);
  }
}


void DataSenderMonitorCollection::addDQMEventSample(I2OChain const& i2oChain)
{
  // sanity checks
  if (i2oChain.messageCode() != Header::DQM_EVENT) {return;}
  if (! i2oChain.complete()) {return;}

  // fetch basic data from the I2OChain
  double eventSize = static_cast<double>(i2oChain.totalDataSize());

  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getRBRecordPointer(i2oChain, rbRecordPtr);
    if (pointersAreValid)
    {
      pointersAreValid = getFURecordPointer(i2oChain, rbRecordPtr, fuRecordPtr);
    }
  }

  // accumulate the data of interest
  if (pointersAreValid)
  {
    rbRecordPtr->dqmEventSize.addSample(eventSize);
    fuRecordPtr->dqmEventSize.addSample(eventSize);
  }
}


void DataSenderMonitorCollection::addErrorEventSample(I2OChain const& i2oChain)
{
  // sanity checks
  if (i2oChain.messageCode() != Header::ERROR_EVENT) {return;}
  if (! i2oChain.complete()) {return;}

  // fetch basic data from the I2OChain
  double eventSize = static_cast<double>(i2oChain.totalDataSize());

  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getRBRecordPointer(i2oChain, rbRecordPtr);
    if (pointersAreValid)
    {
      pointersAreValid = getFURecordPointer(i2oChain, rbRecordPtr, fuRecordPtr);
    }
  }

  // accumulate the data of interest
  if (pointersAreValid)
  {
    rbRecordPtr->errorEventSize.addSample(eventSize);
    fuRecordPtr->errorEventSize.addSample(eventSize);
  }
}


void DataSenderMonitorCollection::addFaultyEventSample(I2OChain const& i2oChain)
{
  // fetch basic data from the I2OChain
  double eventSize = static_cast<double>(i2oChain.totalDataSize());

  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getRBRecordPointer(i2oChain, rbRecordPtr);
    if (pointersAreValid)
    {
      pointersAreValid = getFURecordPointer(i2oChain, rbRecordPtr, fuRecordPtr);
    }
  }

  // accumulate the data of interest
  if (pointersAreValid)
  {
    if (i2oChain.messageCode() == Header::DQM_EVENT)
    {
      rbRecordPtr->faultyDQMEventSize.addSample(eventSize);
      fuRecordPtr->faultyDQMEventSize.addSample(eventSize);
    }
    else
    {
      rbRecordPtr->faultyEventSize.addSample(eventSize);
      fuRecordPtr->faultyEventSize.addSample(eventSize);
    }
  }
}


void DataSenderMonitorCollection::incrementDataDiscardCount(I2OChain const& i2oChain)
{
  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getRBRecordPointer(i2oChain, rbRecordPtr);
    if (pointersAreValid)
    {
      pointersAreValid = getFURecordPointer(i2oChain, rbRecordPtr, fuRecordPtr);
    }
  }

  // accumulate the data of interest
  if (pointersAreValid)
  {
    rbRecordPtr->dataDiscardCount.addSample(1);
    fuRecordPtr->dataDiscardCount.addSample(1);
  }
}


void DataSenderMonitorCollection::incrementDQMDiscardCount(I2OChain const& i2oChain)
{
  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getRBRecordPointer(i2oChain, rbRecordPtr);
    if (pointersAreValid)
    {
      pointersAreValid = getFURecordPointer(i2oChain, rbRecordPtr, fuRecordPtr);
    }
  }

  // accumulate the data of interest
  if (pointersAreValid)
  {
    rbRecordPtr->dqmDiscardCount.addSample(1);
    fuRecordPtr->dqmDiscardCount.addSample(1);
  }
}


void DataSenderMonitorCollection::incrementSkippedDiscardCount(I2OChain const& i2oChain)
{
  // look up the monitoring records that we need
  bool pointersAreValid;
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    pointersAreValid = getRBRecordPointer(i2oChain, rbRecordPtr);
    if (pointersAreValid)
    {
      pointersAreValid = getFURecordPointer(i2oChain, rbRecordPtr, fuRecordPtr);
    }
  }

  // accumulate the data of interest
  if (pointersAreValid)
  {
    rbRecordPtr->skippedDiscardCount.addSample(1);
    fuRecordPtr->skippedDiscardCount.addSample(1);
  }
}


DataSenderMonitorCollection::OutputModuleResultsList
DataSenderMonitorCollection::getTopLevelOutputModuleResults() const
{
  boost::mutex::scoped_lock sl(_collectionsMutex);

  return buildOutputModuleResults(_outputModuleMap);
}


DataSenderMonitorCollection::ResourceBrokerResultsList
DataSenderMonitorCollection::getAllResourceBrokerResults() const
{
  boost::mutex::scoped_lock sl(_collectionsMutex);
  ResourceBrokerResultsList resultsList;

  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapIter;
  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapEnd =
    _resourceBrokerMap.end();
  for (rbMapIter = _resourceBrokerMap.begin(); rbMapIter != rbMapEnd; ++rbMapIter)
    {
      RBRecordPtr rbRecordPtr = rbMapIter->second;
      RBResultPtr result = buildResourceBrokerResult(rbRecordPtr);
      result->uniqueRBID = rbMapIter->first;
      resultsList.push_back(result);
    }

  return resultsList;
}


DataSenderMonitorCollection::RBResultPtr
DataSenderMonitorCollection::getOneResourceBrokerResult(UniqueResourceBrokerID_t uniqueRBID) const
{
  boost::mutex::scoped_lock sl(_collectionsMutex);
  RBResultPtr result;

  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapIter;
  rbMapIter = _resourceBrokerMap.find(uniqueRBID);
  if (rbMapIter != _resourceBrokerMap.end())
    {
      RBRecordPtr rbRecordPtr = rbMapIter->second;
      result = buildResourceBrokerResult(rbRecordPtr);
      result->uniqueRBID = rbMapIter->first;
    }

  return result;
}


DataSenderMonitorCollection::OutputModuleResultsList
DataSenderMonitorCollection::getOutputModuleResultsForRB(UniqueResourceBrokerID_t uniqueRBID) const
{
  boost::mutex::scoped_lock sl(_collectionsMutex);
  OutputModuleResultsList resultsList;

  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapIter;
  rbMapIter = _resourceBrokerMap.find(uniqueRBID);
  if (rbMapIter != _resourceBrokerMap.end())
    {
      RBRecordPtr rbRecordPtr = rbMapIter->second;
      resultsList = buildOutputModuleResults(rbRecordPtr->outputModuleMap);
    }

  return resultsList;
}


DataSenderMonitorCollection::FilterUnitResultsList
DataSenderMonitorCollection::getFilterUnitResultsForRB(UniqueResourceBrokerID_t uniqueRBID) const
{
  boost::mutex::scoped_lock sl(_collectionsMutex);
  FilterUnitResultsList resultsList;

  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapIter;
  rbMapIter = _resourceBrokerMap.find(uniqueRBID);
  if (rbMapIter != _resourceBrokerMap.end())
    {
      RBRecordPtr rbRecordPtr = rbMapIter->second;
      std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapIter;
      std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapEnd =
        rbRecordPtr->filterUnitMap.end();        
      for (fuMapIter = rbRecordPtr->filterUnitMap.begin();
           fuMapIter != fuMapEnd; ++fuMapIter)
        {
          FURecordPtr fuRecordPtr = fuMapIter->second;
          FUResultPtr result(new FilterUnitResult(fuRecordPtr->key));
          result->initMsgCount = fuRecordPtr->initMsgCount;
          result->lastRunNumber = fuRecordPtr->lastRunNumber;
          result->lastEventNumber = fuRecordPtr->lastEventNumber;
          fuRecordPtr->shortIntervalEventSize.getStats(result->shortIntervalEventStats);
          fuRecordPtr->mediumIntervalEventSize.getStats(result->mediumIntervalEventStats);
          fuRecordPtr->dqmEventSize.getStats(result->dqmEventStats);
          fuRecordPtr->errorEventSize.getStats(result->errorEventStats);
          fuRecordPtr->faultyEventSize.getStats(result->faultyEventStats);
          fuRecordPtr->faultyDQMEventSize.getStats(result->faultyDQMEventStats);
          fuRecordPtr->dataDiscardCount.getStats(result->dataDiscardStats);
          fuRecordPtr->dqmDiscardCount.getStats(result->dqmDiscardStats);
          fuRecordPtr->skippedDiscardCount.getStats(result->skippedDiscardStats);

          result->outstandingDataDiscardCount =
            result->initMsgCount +
            result->shortIntervalEventStats.getSampleCount() +
            result->errorEventStats.getSampleCount() +
            result->faultyEventStats.getSampleCount() -
            result->dataDiscardStats.getSampleCount();
          result->outstandingDQMDiscardCount =
            result->dqmEventStats.getSampleCount() +
            result->faultyDQMEventStats.getSampleCount() -
            result->dqmDiscardStats.getSampleCount();

          resultsList.push_back(result);
        }
    }

  return resultsList;
}


void DataSenderMonitorCollection::do_calculateStatistics()
{
  boost::mutex::scoped_lock sl(_collectionsMutex);

  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapIter;
  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapEnd =
    _resourceBrokerMap.end();
  for (rbMapIter=_resourceBrokerMap.begin(); rbMapIter!=rbMapEnd; ++rbMapIter)
    {
      RBRecordPtr rbRecordPtr = rbMapIter->second;
      rbRecordPtr->eventSize.calculateStatistics();
      rbRecordPtr->dqmEventSize.calculateStatistics();
      rbRecordPtr->errorEventSize.calculateStatistics();
      rbRecordPtr->faultyEventSize.calculateStatistics();
      rbRecordPtr->faultyDQMEventSize.calculateStatistics();
      rbRecordPtr->dataDiscardCount.calculateStatistics();
      rbRecordPtr->dqmDiscardCount.calculateStatistics();
      rbRecordPtr->skippedDiscardCount.calculateStatistics();
      calcStatsForOutputModules(rbRecordPtr->outputModuleMap);

      std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapIter;
      std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapEnd =
        rbRecordPtr->filterUnitMap.end();        
      for (fuMapIter = rbRecordPtr->filterUnitMap.begin();
           fuMapIter != fuMapEnd; ++fuMapIter)
        {
          FURecordPtr fuRecordPtr = fuMapIter->second;
          fuRecordPtr->shortIntervalEventSize.calculateStatistics();
          fuRecordPtr->mediumIntervalEventSize.calculateStatistics();
          fuRecordPtr->dqmEventSize.calculateStatistics();
          fuRecordPtr->errorEventSize.calculateStatistics();
          fuRecordPtr->faultyEventSize.calculateStatistics();
          fuRecordPtr->faultyDQMEventSize.calculateStatistics();
          fuRecordPtr->dataDiscardCount.calculateStatistics();
          fuRecordPtr->dqmDiscardCount.calculateStatistics();
          fuRecordPtr->skippedDiscardCount.calculateStatistics();
          calcStatsForOutputModules(fuRecordPtr->outputModuleMap);
        }
    }

  calcStatsForOutputModules(_outputModuleMap);
}


void DataSenderMonitorCollection::do_reset()
{
  boost::mutex::scoped_lock sl(_collectionsMutex);

  _connectedRBs = 0;
  _connectedEPs = 0;
  _activeEPs = 0;
  _outstandingDataDiscards = 0;
  _outstandingDQMDiscards = 0;
  _faultyEvents = 0;
  _ignoredDiscards = 0;
  _resourceBrokerMap.clear();
  _outputModuleMap.clear();
}


void DataSenderMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("connectedRBs", &_connectedRBs));
  infoSpaceItems.push_back(std::make_pair("connectedEPs", &_connectedEPs));
  infoSpaceItems.push_back(std::make_pair("activeEPs", &_activeEPs));
  infoSpaceItems.push_back(std::make_pair("outstandingDataDiscards", &_outstandingDataDiscards));
  infoSpaceItems.push_back(std::make_pair("outstandingDQMDiscards", &_outstandingDQMDiscards));
  infoSpaceItems.push_back(std::make_pair("faultyEvents", &_faultyEvents));
  infoSpaceItems.push_back(std::make_pair("ignoredDiscards", &_ignoredDiscards));
}


void DataSenderMonitorCollection::do_updateInfoSpaceItems()
{
  boost::mutex::scoped_lock sl(_collectionsMutex);

  _connectedRBs = static_cast<xdata::UnsignedInteger32>(_resourceBrokerMap.size());

  uint32_t localEPCount = 0;
  uint32_t localActiveEPCount = 0;
  int localMissingDataDiscardCount = 0;
  int localMissingDQMDiscardCount = 0;
  uint32_t localFaultyEventsCount = 0;
  uint32_t localIgnoredDiscardCount = 0;
  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapIter;
  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapEnd =
    _resourceBrokerMap.end();
  for (rbMapIter = _resourceBrokerMap.begin(); rbMapIter != rbMapEnd; ++rbMapIter)
  {
    RBRecordPtr rbRecordPtr = rbMapIter->second;
    localEPCount += rbRecordPtr->filterUnitMap.size();

    MonitoredQuantity::Stats skippedDiscardStats;
    rbRecordPtr->skippedDiscardCount.getStats(skippedDiscardStats);
    localIgnoredDiscardCount += skippedDiscardStats.getSampleCount();

    MonitoredQuantity::Stats eventStats;
    MonitoredQuantity::Stats errorEventStats;
    MonitoredQuantity::Stats dataDiscardStats;
    rbRecordPtr->eventSize.getStats(eventStats);
    rbRecordPtr->errorEventSize.getStats(errorEventStats);
    rbRecordPtr->dataDiscardCount.getStats(dataDiscardStats);
    localMissingDataDiscardCount += rbRecordPtr->initMsgCount + eventStats.getSampleCount() +
      errorEventStats.getSampleCount() - dataDiscardStats.getSampleCount();

    MonitoredQuantity::Stats dqmEventStats;
    MonitoredQuantity::Stats dqmDiscardStats;
    rbRecordPtr->dqmEventSize.getStats(dqmEventStats);
    rbRecordPtr->dqmDiscardCount.getStats(dqmDiscardStats);
    localMissingDQMDiscardCount += dqmEventStats.getSampleCount() -
      dqmDiscardStats.getSampleCount();

    MonitoredQuantity::Stats faultyEventStats;
    rbRecordPtr->faultyEventSize.getStats(faultyEventStats);
    localFaultyEventsCount += faultyEventStats.getSampleCount();
    MonitoredQuantity::Stats faultyDQMEventStats;
    rbRecordPtr->faultyDQMEventSize.getStats(faultyDQMEventStats);
    localFaultyEventsCount += faultyDQMEventStats.getSampleCount();

    std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapIter;
    std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapEnd =
      rbRecordPtr->filterUnitMap.end();        
    for (fuMapIter = rbRecordPtr->filterUnitMap.begin(); fuMapIter != fuMapEnd; ++fuMapIter)
    {
      FURecordPtr fuRecordPtr = fuMapIter->second;
      MonitoredQuantity::Stats fuMediumIntervalEventStats;
      fuRecordPtr->mediumIntervalEventSize.getStats(fuMediumIntervalEventStats);
      if (fuMediumIntervalEventStats.getSampleCount(MonitoredQuantity::RECENT) > 0) {
        ++localActiveEPCount;
      }
    }
  }
  _connectedEPs = static_cast<xdata::UnsignedInteger32>(localEPCount);
  _activeEPs = static_cast<xdata::UnsignedInteger32>(localActiveEPCount);
  _outstandingDataDiscards = static_cast<xdata::Integer32>(localMissingDataDiscardCount);
  _outstandingDQMDiscards = static_cast<xdata::Integer32>(localMissingDQMDiscardCount);
  _faultyEvents = static_cast<xdata::UnsignedInteger32>(localFaultyEventsCount);
  _ignoredDiscards = static_cast<xdata::UnsignedInteger32>(localIgnoredDiscardCount);

  faultyEventsAlarm(localFaultyEventsCount);
  ignoredDiscardAlarm(localIgnoredDiscardCount);
}


void DataSenderMonitorCollection::faultyEventsAlarm(const uint32_t& faultyEventsCount) const
{
  const std::string alarmName = "FaultyEvents";

  if (faultyEventsCount > 0)
  {
    std::ostringstream msg;
    msg << "Missing or faulty I2O fragments for " <<
      faultyEventsCount <<
      " events. These events are lost!";
    XCEPT_DECLARE(stor::exception::FaultyEvents, ex, msg.str());
    _alarmHandler->raiseAlarm(alarmName, AlarmHandler::ERROR, ex);
  }
  else
  {
    _alarmHandler->revokeAlarm(alarmName);
  }
}


void DataSenderMonitorCollection::ignoredDiscardAlarm(const uint32_t& ignoredDiscardCount) const
{
  const std::string alarmName = "IgnoredDiscard";

  if ( ignoredDiscardCount > 0)
  {
    std::ostringstream msg;
    msg << ignoredDiscardCount <<
      " discard messages ignored. These events might be stuck in the resource broker.";
    XCEPT_DECLARE(stor::exception::IgnoredDiscard, ex, msg.str());
    _alarmHandler->raiseAlarm(alarmName, AlarmHandler::ERROR, ex);
  }
  else
  {
    _alarmHandler->revokeAlarm(alarmName);
  }
}


typedef DataSenderMonitorCollection DSMC;

bool DSMC::getAllNeededPointers(I2OChain const& i2oChain,
                                DSMC::RBRecordPtr& rbRecordPtr,
                                DSMC::FURecordPtr& fuRecordPtr,
                                DSMC::OutModRecordPtr& topLevelOutModPtr,
                                DSMC::OutModRecordPtr& rbSpecificOutModPtr,
                                DSMC::OutModRecordPtr& fuSpecificOutModPtr)
{
  ResourceBrokerKey rbKey(i2oChain);
  if (! rbKey.isValid) {return false;}
  FilterUnitKey fuKey(i2oChain);
  if (! fuKey.isValid) {return false;}
  OutputModuleKey outModKey = i2oChain.outputModuleId();

  topLevelOutModPtr = getOutputModuleRecord(_outputModuleMap, outModKey);

  rbRecordPtr = getResourceBrokerRecord(rbKey);
  rbSpecificOutModPtr = getOutputModuleRecord(rbRecordPtr->outputModuleMap,
                                              outModKey);

  fuRecordPtr = getFilterUnitRecord(rbRecordPtr, fuKey);
  fuSpecificOutModPtr = getOutputModuleRecord(fuRecordPtr->outputModuleMap,
                                              outModKey);

  return true;
}


bool DSMC::getRBRecordPointer(I2OChain const& i2oChain,
                              DSMC::RBRecordPtr& rbRecordPtr)
{
  ResourceBrokerKey rbKey(i2oChain);
  if (! rbKey.isValid) {return false;}

  rbRecordPtr = getResourceBrokerRecord(rbKey);
  return true;
}


bool DSMC::getFURecordPointer(I2OChain const& i2oChain,
                              DSMC::RBRecordPtr& rbRecordPtr,
                              DSMC::FURecordPtr& fuRecordPtr)
{
  FilterUnitKey fuKey(i2oChain);
  if (! fuKey.isValid) {return false;}

  fuRecordPtr = getFilterUnitRecord(rbRecordPtr, fuKey);
  return true;
}


DSMC::RBRecordPtr
DSMC::getResourceBrokerRecord(DSMC::ResourceBrokerKey const& rbKey)
{
  RBRecordPtr rbRecordPtr;
  UniqueResourceBrokerID_t uniqueRBID = getUniqueResourceBrokerID(rbKey);
  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapIter;
  rbMapIter = _resourceBrokerMap.find(uniqueRBID);
  if (rbMapIter == _resourceBrokerMap.end())
    {
      rbRecordPtr.reset(new ResourceBrokerRecord(rbKey,_updateInterval));
      _resourceBrokerMap[uniqueRBID] = rbRecordPtr;
    }
  else
    {
      rbRecordPtr = rbMapIter->second;
    }
  return rbRecordPtr;
}


DSMC::UniqueResourceBrokerID_t
DSMC::getUniqueResourceBrokerID(DSMC::ResourceBrokerKey const& rbKey)
{
  UniqueResourceBrokerID_t uniqueID;
  std::map<ResourceBrokerKey, UniqueResourceBrokerID_t>::const_iterator rbMapIter;
  rbMapIter = _resourceBrokerIDs.find(rbKey);
  if (rbMapIter == _resourceBrokerIDs.end())
    {
      std::string workString = rbKey.hltURL +
        boost::lexical_cast<std::string>(rbKey.hltTid) +
        boost::lexical_cast<std::string>(rbKey.hltInstance) +
        boost::lexical_cast<std::string>(rbKey.hltLocalId) +
        rbKey.hltClassName;
      uLong crc = crc32(0L, Z_NULL, 0);
      Bytef* crcbuf = (Bytef*) workString.data();
      crc = crc32(crc, crcbuf, workString.length());
      uniqueID = static_cast<UniqueResourceBrokerID_t>(crc);
      _resourceBrokerIDs[rbKey] = uniqueID;
    }
  else
    {
      uniqueID = rbMapIter->second;
    }
  return uniqueID;
}


DSMC::FURecordPtr
DSMC::getFilterUnitRecord(DSMC::RBRecordPtr& rbRecordPtr,
                          DSMC::FilterUnitKey const& fuKey)
{
  FURecordPtr fuRecordPtr;
  std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapIter;
  fuMapIter = rbRecordPtr->filterUnitMap.find(fuKey);
  if (fuMapIter == rbRecordPtr->filterUnitMap.end())
    {
      fuRecordPtr.reset(new FilterUnitRecord(fuKey,_updateInterval));
      rbRecordPtr->filterUnitMap[fuKey] = fuRecordPtr;
    }
  else
    {
      fuRecordPtr = fuMapIter->second;
    }
  return fuRecordPtr;
}


DSMC::OutModRecordPtr
DSMC::getOutputModuleRecord(OutputModuleRecordMap& outModMap,
                            DSMC::OutputModuleKey const& outModKey)
{
  OutModRecordPtr outModRecordPtr;
  OutputModuleRecordMap::const_iterator omMapIter;
  omMapIter = outModMap.find(outModKey);
  if (omMapIter == outModMap.end())
    {
      outModRecordPtr.reset(new OutputModuleRecord(_updateInterval));

      outModRecordPtr->name = "Unknown";
      outModRecordPtr->id = outModKey;
      outModRecordPtr->initMsgSize = 0;

      outModMap[outModKey] = outModRecordPtr;
    }
  else
    {
      outModRecordPtr = omMapIter->second;
    }
  return outModRecordPtr;
}


DSMC::OutputModuleResultsList
DSMC::buildOutputModuleResults(DSMC::OutputModuleRecordMap const& outputModuleMap) const
{
  OutputModuleResultsList resultsList;

  OutputModuleRecordMap::const_iterator omMapIter;
  OutputModuleRecordMap::const_iterator omMapEnd = outputModuleMap.end();
  for (omMapIter = outputModuleMap.begin(); omMapIter != omMapEnd; ++omMapIter)
    {
      OutModRecordPtr outModRecordPtr = omMapIter->second;
      boost::shared_ptr<OutputModuleResult> result(new OutputModuleResult());
      result->name = outModRecordPtr->name;
      result->id = outModRecordPtr->id;
      result->initMsgSize = outModRecordPtr->initMsgSize;
      outModRecordPtr->eventSize.getStats(result->eventStats);
      resultsList.push_back(result);
    }

  return resultsList;
}


DSMC::RBResultPtr
DSMC::buildResourceBrokerResult(DSMC::RBRecordPtr const& rbRecordPtr) const
{
  RBResultPtr result(new ResourceBrokerResult(rbRecordPtr->key));

  result->filterUnitCount = rbRecordPtr->filterUnitMap.size();
  result->initMsgCount = rbRecordPtr->initMsgCount;
  result->lastRunNumber = rbRecordPtr->lastRunNumber;
  result->lastEventNumber = rbRecordPtr->lastEventNumber;
  rbRecordPtr->eventSize.getStats(result->eventStats);
  rbRecordPtr->dqmEventSize.getStats(result->dqmEventStats);
  rbRecordPtr->errorEventSize.getStats(result->errorEventStats);
  rbRecordPtr->faultyEventSize.getStats(result->faultyEventStats);
  rbRecordPtr->faultyDQMEventSize.getStats(result->faultyDQMEventStats);
  rbRecordPtr->dataDiscardCount.getStats(result->dataDiscardStats);
  rbRecordPtr->dqmDiscardCount.getStats(result->dqmDiscardStats);
  rbRecordPtr->skippedDiscardCount.getStats(result->skippedDiscardStats);

  result->outstandingDataDiscardCount =
    result->initMsgCount +
    result->eventStats.getSampleCount() +
    result->errorEventStats.getSampleCount() +
    result->faultyEventStats.getSampleCount() -
    result->dataDiscardStats.getSampleCount();
  result->outstandingDQMDiscardCount =
    result->dqmEventStats.getSampleCount() +
    result->faultyDQMEventStats.getSampleCount() -
    result->dqmDiscardStats.getSampleCount();

  return result;
}


void DSMC::calcStatsForOutputModules(DSMC::OutputModuleRecordMap& outputModuleMap)
{
  OutputModuleRecordMap::const_iterator omMapIter;
  OutputModuleRecordMap::const_iterator omMapEnd = outputModuleMap.end();
  for (omMapIter = outputModuleMap.begin(); omMapIter != omMapEnd; ++omMapIter)
    {
      OutModRecordPtr outModRecordPtr = omMapIter->second;

      //outModRecordPtr->fragmentSize.calculateStatistics();
      outModRecordPtr->eventSize.calculateStatistics();
    }
}


bool stor::compareRBResultPtrValues(DSMC::RBResultPtr firstValue,
                                    DSMC::RBResultPtr secondValue)
{
  return *firstValue < *secondValue;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
