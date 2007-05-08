//
// File: EventFilter/StorageManager/src/DQMServiceManager.cc
//
// (W.Badgett)
//
// $Id$
//

#include "FWCore/Utilities/interface/DebugMacros.h"
#include <EventFilter/StorageManager/interface/DQMServiceManager.h>
#include <FWCore/Utilities/interface/Exception.h>
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"
#include "TROOT.h"
#include "TApplication.h"

using namespace edm;
using namespace std;
using namespace stor;
using boost::shared_ptr;


DQMServiceManager::DQMServiceManager(std::string filePrefix,
				     int purgeTime,
				     int readyTime,
                                     bool collateDQM):
  collateDQM_(collateDQM),
  runNumber_(-1),
  lumiSection_(-1),
  instance_(-1),
  nUpdates_(0),
  filePrefix_(filePrefix),
  purgeTime_(purgeTime),
  readyTime_(readyTime)
{
  dqmInstances_.reserve(20);

  gROOT->SetBatch(kTRUE);

  int argc = 0;
  rootApplication_ = new TApplication("DQMServiceManager", &argc, NULL);
} 

DQMServiceManager::~DQMServiceManager()
{ stop();}

void DQMServiceManager::stop()
{ if(collateDQM_) writeAndPurgeDQMInstances(true);}

void DQMServiceManager::manageDQMEventMsg(DQMEventMsgView& msg)
{
  // At the moment implement the behaviour such that collateDQM = archive DQM
  // so if we don't collate we also don't archive, else we need changes here
  if(!collateDQM_) {
    // no collation just pass the DQMEvent to the Event Server and return
    if (DQMeventServer_.get() != NULL) {
      DQMeventServer_->processDQMEvent(msg);
    }
    return;
  }

  if ( (int)msg.runNumber() > runNumber_ )
  {
    FDEBUG(1) << "DQMServiceManager found new run " << 
	msg.runNumber() << std::endl;
    runNumber_   = msg.runNumber();
    lumiSection_ = msg.lumiSection();
    instance_    = msg.updateNumber();
  }
  else if ((int)msg.lumiSection() > lumiSection_ )
  {
    FDEBUG(2) << "DQMServiceManager found new lumiSection " << 
      msg.lumiSection() << " run " <<
      msg.runNumber() << std::endl;
    lumiSection_ = msg.lumiSection();
    instance_    = msg.updateNumber();
  }
  else if ( (int)msg.updateNumber() > instance_ )
  {
    FDEBUG(3) << "DQMServiceManager found new instance " << 
      msg.updateNumber() << " lumiSection " <<
      msg.lumiSection()  << " run " <<
      msg.runNumber()    << std::endl;
    instance_    = msg.updateNumber();
  }

  DQMInstance * dqm = findDQMInstance(msg.runNumber(),
				      msg.lumiSection(),
				      msg.updateNumber());
  if ( dqm == NULL ) 
  {
    dqm = new DQMInstance(msg.runNumber(),
			  msg.lumiSection(),
			  msg.updateNumber(),
			  purgeTime_,
			  readyTime_);
    dqmInstances_.push_back(dqm);
    int preSize = dqmInstances_.size();

    // At this point, purge old instances from the vector
    writeAndPurgeDQMInstances(false);
    FDEBUG(4) << "Live DQMInstances before purge " << preSize << 
      " and after " << dqmInstances_.size() << std::endl;
  }

  edm::StreamDQMDeserializer deserializer;
  std::auto_ptr<DQMEvent::TObjectTable> toTablePtr =
    deserializer.deserializeDQMEvent(msg);

  DQMEvent::TObjectTable::const_iterator toIter;
  for (toIter = toTablePtr->begin();
       toIter != toTablePtr->end(); toIter++) 
  {
    std::string subFolderName = toIter->first;
    std::vector<TObject *> toList = toIter->second;

    for (int tdx = 0; tdx < (int) toList.size(); tdx++) 
    {
      TObject *object = toList[tdx];
      dqm->updateObject(msg.topFolderName(),
			subFolderName,
			object);
      delete(object);
    }
  }
}

DQMInstance * DQMServiceManager::findDQMInstance(int runNumber_, 
						 int lumiSection_, 
						 int instance_)
{
  DQMInstance * reply = NULL;
  int n = dqmInstances_.size();
  for ( int i=0; (i<n) && (reply==NULL); i++)
  {
    if( dqmInstances_[i] != NULL )
    {
      if ( ( dqmInstances_[i]->getRunNumber()   == runNumber_ ) && 
	   ( dqmInstances_[i]->getLumiSection() == lumiSection_ ) && 
	   ( dqmInstances_[i]->getInstance()    == instance_ ) )
      { reply = dqmInstances_[i]; }
    }
  }
  return(reply);
}

int DQMServiceManager::writeAndPurgeDQMInstances(bool writeAll)
{
  int reply = 0;

  TTimeStamp now;
  now.Set();

  int minInstances = 1;
  if ( writeAll ) { minInstances=0;}

  // Always keep at least one instance in memory

  int n = dqmInstances_.size();
  for ( std::vector<DQMInstance *>::iterator i0 = dqmInstances_.begin(); 
	(i0 != dqmInstances_.end()) && (n>minInstances); ++i0)
  {
    DQMInstance * instance = *i0;
    if ( instance != NULL )
    {
      if ( instance->isReady(now.GetSec()) || writeAll)
      {
	instance->writeFile(filePrefix_);
	delete(instance);
	reply++;
        // HWKC: this looks dangerous to me, is the iterator invalidated?
	dqmInstances_.erase(i0);
      }
    }
    n = dqmInstances_.size();
  }
  return(reply);
}

DQMInstance * DQMServiceManager::getLastDQMInstance()
{
  DQMInstance * instance = NULL;
  if ( dqmInstances_.size() > 0 ) { instance = dqmInstances_.back(); }
  return(instance);
}
