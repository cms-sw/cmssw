// $Id$

#include <string>
#include <sstream>
#include <iomanip>

#include <zlib.h>
#include <boost/lexical_cast.hpp>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"

using namespace stor;


DataSenderMonitorCollection::DataSenderMonitorCollection(xdaq::Application *app) :
MonitorCollection(app),
_connectedRBs(0)
{
  _infoSpaceItems.push_back(std::make_pair("connectedRBs", &_connectedRBs));

  putItemsIntoInfoSpace();
}


void DataSenderMonitorCollection::addFragmentSample(I2OChain const& i2oChain)
{
  // focus only on INIT and event fragments, for now
  if (i2oChain.messageCode() != Header::INIT &&
      i2oChain.messageCode() != Header::EVENT) {return;}
  if (i2oChain.fragmentCount() != 1) {return;}

  // fetch basic data from the I2OChain
  //double fragmentSize = static_cast<double>(i2oChain.totalDataSize());

  // look up the monitoring records that we need
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  OutModRecordPtr topLevelOutModPtr, rbSpecificOutModPtr, fuSpecificOutModPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    getAllNeededPointers(i2oChain, rbRecordPtr, fuRecordPtr,
                         topLevelOutModPtr, rbSpecificOutModPtr,
                         fuSpecificOutModPtr);
  }

  // accumulate the data of interest
  //topLevelOutModPtr->fragmentSize.addSample(fragmentSize);
  //rbSpecificOutModPtr->fragmentSize.addSample(fragmentSize);
  //fuSpecificOutModPtr->fragmentSize.addSample(fragmentSize);
}


void DataSenderMonitorCollection::addInitSample(I2OChain const& i2oChain)
{
  // sanity checks
  if (i2oChain.messageCode() != Header::INIT) {return;}
  if (! i2oChain.complete()) {return;}

  // fetch basic data from the I2OChain
  std::string outModName = i2oChain.outputModuleLabel();
  unsigned int msgSize = i2oChain.totalDataSize();

  // look up the monitoring records that we need
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  OutModRecordPtr topLevelOutModPtr, rbSpecificOutModPtr, fuSpecificOutModPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    getAllNeededPointers(i2oChain, rbRecordPtr, fuRecordPtr,
                         topLevelOutModPtr, rbSpecificOutModPtr,
                         fuSpecificOutModPtr);
  }

  // accumulate the data of interest
  topLevelOutModPtr->name = outModName;
  topLevelOutModPtr->initMsgSize = msgSize;

  ++rbRecordPtr->initMsgCount;
  rbSpecificOutModPtr->name = outModName;
  rbSpecificOutModPtr->initMsgSize = msgSize;

  ++fuRecordPtr->initMsgCount;
  fuSpecificOutModPtr->name = outModName;
  fuSpecificOutModPtr->initMsgSize = msgSize;
}


void DataSenderMonitorCollection::addEventSample(I2OChain const& i2oChain)
{
  // sanity checks
  if (i2oChain.messageCode() != Header::EVENT) {return;}
  if (! i2oChain.complete()) {return;}

  // fetch basic data from the I2OChain
  double eventSize = static_cast<double>(i2oChain.totalDataSize());
  unsigned int runNumber = i2oChain.runNumber();
  unsigned int eventNumber = i2oChain.eventNumber();

  // look up the monitoring records that we need
  RBRecordPtr rbRecordPtr;
  FURecordPtr fuRecordPtr;
  OutModRecordPtr topLevelOutModPtr, rbSpecificOutModPtr, fuSpecificOutModPtr;
  {
    boost::mutex::scoped_lock sl(_collectionsMutex);
    getAllNeededPointers(i2oChain, rbRecordPtr, fuRecordPtr,
                         topLevelOutModPtr, rbSpecificOutModPtr,
                         fuSpecificOutModPtr);
  }

  // accumulate the data of interest
  topLevelOutModPtr->eventSize.addSample(eventSize);

  rbRecordPtr->lastRunNumber = runNumber;
  rbRecordPtr->lastEventNumber = eventNumber;
  rbRecordPtr->eventSize.addSample(eventSize);
  rbSpecificOutModPtr->eventSize.addSample(eventSize);

  fuRecordPtr->lastRunNumber = runNumber;
  fuRecordPtr->lastEventNumber = eventNumber;
  fuRecordPtr->eventSize.addSample(eventSize);
  fuSpecificOutModPtr->eventSize.addSample(eventSize);
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
          fuRecordPtr->eventSize.getStats(result->eventStats);
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
      calcStatsForOutputModules(rbRecordPtr->outputModuleMap);

      std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapIter;
      std::map<FilterUnitKey, FURecordPtr>::const_iterator fuMapEnd =
        rbRecordPtr->filterUnitMap.end();        
      for (fuMapIter = rbRecordPtr->filterUnitMap.begin();
           fuMapIter != fuMapEnd; ++fuMapIter)
        {
          FURecordPtr fuRecordPtr = fuMapIter->second;
          fuRecordPtr->eventSize.calculateStatistics();
          calcStatsForOutputModules(fuRecordPtr->outputModuleMap);
        }
    }

  calcStatsForOutputModules(_outputModuleMap);
}


void DataSenderMonitorCollection::do_updateInfoSpace()
{
  boost::mutex::scoped_lock sl(_collectionsMutex);

  std::string errorMsg =
    "Failed to update values of items in info space " + _infoSpace->name();

  // Lock the infospace to assure that all items are consistent
  try
  {
    _infoSpace->lock();

    _connectedRBs = static_cast<xdata::UnsignedInteger32>(_resourceBrokerMap.size());

    _infoSpace->unlock();
  }
  catch(std::exception &e)
  {
    _infoSpace->unlock();
 
    errorMsg += ": ";
    errorMsg += e.what();
    XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
  }
  catch (...)
  {
    _infoSpace->unlock();
 
    errorMsg += " : unknown exception";
    XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
  }

  try
  {
    // The fireItemGroupChanged locks the infospace
    _infoSpace->fireItemGroupChanged(_infoSpaceItemNames, this);
  }
  catch (xdata::exception::Exception &e)
  {
    XCEPT_RETHROW(stor::exception::Monitoring, errorMsg, e);
  }
}


void DataSenderMonitorCollection::do_reset()
{
  boost::mutex::scoped_lock sl(_collectionsMutex);

  _connectedRBs = 0;
  _resourceBrokerMap.clear();
  _outputModuleMap.clear();
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


DSMC::RBRecordPtr
DSMC::getResourceBrokerRecord(DSMC::ResourceBrokerKey const& rbKey)
{
  RBRecordPtr rbRecordPtr;
  UniqueResourceBrokerID_t uniqueRBID = getUniqueResourceBrokerID(rbKey);
  std::map<UniqueResourceBrokerID_t, RBRecordPtr>::const_iterator rbMapIter;
  rbMapIter = _resourceBrokerMap.find(uniqueRBID);
  if (rbMapIter == _resourceBrokerMap.end())
    {
      rbRecordPtr.reset(new ResourceBrokerRecord(rbKey));
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
      fuRecordPtr.reset(new FilterUnitRecord(fuKey));
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
      outModRecordPtr.reset(new OutputModuleRecord());

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
