// $Id: DQMEventStore.cc,v 1.11 2010/03/09 12:58:04 mommsen Exp $
/// @file: DQMEventStore.cc

#include "TROOT.h"
#include "TTimeStamp.h"

#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventStore.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/Utils.h"

using namespace stor;


DQMEventStore::DQMEventStore(SharedResourcesPtr sr) :
_dqmEventMonColl(sr->_statisticsReporter->getDQMEventMonitorCollection()),
_initMsgColl(sr->_initMsgCollection)
{
  gROOT->SetBatch(kTRUE);
}

DQMEventStore::~DQMEventStore()
{
  clear();
}

void DQMEventStore::clear()
{
  _store.clear();
  while ( ! _recordsReadyToServe.empty() )
    _recordsReadyToServe.pop();
}

void DQMEventStore::setParameters(DQMProcessingParams const& dqmParams)
{
  clear();
  _dqmParams = dqmParams;
}

void DQMEventStore::addDQMEvent(I2OChain const& dqmEvent)
{
  if ( _dqmParams._collateDQM )
    addDQMEventToStore(dqmEvent);
  else
    addDQMEventToReadyToServe(dqmEvent);
}


bool DQMEventStore::getCompletedDQMGroupRecordIfAvailable
(
  DQMEventRecord::GroupRecord& entry
)
{
  if ( _recordsReadyToServe.empty() ) return false;

  entry = _recordsReadyToServe.front();
  _recordsReadyToServe.pop();

  return true;
}


void DQMEventStore::addDQMEventToStore(I2OChain const& dqmEvent)
{
  const DQMKey newKey = dqmEvent.dqmKey();

  // Use efficientAddOrUpdates pattern suggested by Item 24 of 
  // 'Effective STL' by Scott Meyers
  DQMEventRecordMap::iterator pos = _store.lower_bound(newKey);

  if(pos != _store.end() && !(_store.key_comp()(newKey, pos->first)))
  {
    // key already exists
    pos->second->addDQMEventView( getDQMEventView(dqmEvent) );
  }
  else
  {
    // Use pos as a hint to insert a new record, so it can avoid another lookup
    DQMEventRecordPtr record = makeDQMEventRecord(dqmEvent);
    _store.insert(pos, DQMEventRecordMap::value_type(newKey, record));
    
    // At this point, purge old instances from the list
    writeAndPurgeStaleDQMInstances();
  }

  addNextAvailableDQMGroupToReadyToServe( dqmEvent.topFolderName() );
}


void DQMEventStore::addDQMEventToReadyToServe(I2OChain const& dqmEvent)
{
  DQMEventRecordPtr record =
    makeDQMEventRecord(dqmEvent);

  _recordsReadyToServe.push(
    record->populateAndGetGroup( dqmEvent.topFolderName() )
  );
}


void DQMEventStore::addNextAvailableDQMGroupToReadyToServe(const std::string groupName)
{
  DQMEventRecordPtr record = getNewestReadyDQMEventRecord(groupName);
  
  if ( record )
  {  
    _recordsReadyToServe.push(
      record->populateAndGetGroup( groupName )
    );
  }
}


DQMEventRecordPtr
DQMEventStore::makeDQMEventRecord(I2OChain const& dqmEvent)
{
  DQMEventRecordPtr record(
    new DQMEventRecord(dqmEvent.dqmKey(), _dqmParams, _dqmEventMonColl,
      _initMsgColl->maxMsgCount()) 
  );
  record->setEventConsumerTags( dqmEvent.getDQMEventConsumerTags() );
  record->addDQMEventView( getDQMEventView(dqmEvent) );

  return record;
}


DQMEventMsgView DQMEventStore::getDQMEventView(I2OChain const& dqmEvent)
{
  dqmEvent.copyFragmentsIntoBuffer(_tempEventArea);
  return DQMEventMsgView(&_tempEventArea[0]);
}



DQMEventRecordPtr
DQMEventStore::getNewestReadyDQMEventRecord(const std::string groupName) const
{
  DQMEventRecordPtr readyRecord;
  TTimeStamp now;
  now.Set();
  int maxTime(0);

  for (DQMEventRecordMap::const_iterator it = _store.begin(),
         itEnd = _store.end();
       it != itEnd;
       ++it)
  {
    DQMGroup *group = it->second->getDQMGroup(groupName);
    if ( group && group->isReady( now.GetSec() ) && ! group->wasServedSinceUpdate() )
    {
      TTimeStamp *groupTime = group->getLastUpdate();
      if ( groupTime->GetSec() > maxTime )
      {
        maxTime = groupTime->GetSec();
        readyRecord = it->second;
      }
    }
  }
  
  return readyRecord;  
}


void DQMEventStore::writeAndPurgeStaleDQMInstances()
{
  TTimeStamp now;
  now.Set();
  time_t nowSec = now.GetSec();
  
  for (
    DQMEventRecordMap::iterator it = _store.begin();
    it != _store.end();
  )
  {
    if ( it->second->isReady() || it->second->isStale(nowSec) )
    {
      if ( _dqmParams._archiveDQM &&
        (_dqmParams._archiveIntervalDQM > 0) &&
        ((it->second->getLumiSection() % _dqmParams._archiveIntervalDQM) == 0)
      )
      {
        // The instance is written to file when it is ready and intermediate
        // histograms are written and the lumi section matches the
        // one-in-N archival interval
        it->second->writeFile(_dqmParams._filePrefixDQM, false);
      }
      _store.erase(it++);
    }
    else
    {
      ++it;
    }
  }
}


void DQMEventStore::writeAndPurgeAllDQMInstances()
{
  if ( _dqmParams._archiveDQM )
    writeLatestReadyDQMInstance();

  _store.clear();
}


void DQMEventStore::writeLatestReadyDQMInstance() const
{
  TTimeStamp now;
  now.Set();
  time_t nowSec = now.GetSec();
  
  // Iterate over map in reverse sense. Thus, we encounter the
  // newest instance first
  DQMEventRecordMap::const_reverse_iterator it = _store.rbegin(),
    itEnd = _store.rend();
  while ( it != itEnd )
  {
    if ( it->second->isReady() || it->second->isStale(nowSec) )
    {
      it->second->writeFile(_dqmParams._filePrefixDQM, true);
      break;
    }
    ++it;
  }
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
