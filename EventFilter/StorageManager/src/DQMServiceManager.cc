//
// File: EventFilter/StorageManager/src/DQMServiceManager.cc
//
// (W.Badgett)
//
// $Id: DQMServiceManager.cc,v 1.3 2007/06/11 10:04:49 badgett Exp $
//

#include "FWCore/Utilities/interface/DebugMacros.h"
#include <EventFilter/StorageManager/interface/DQMServiceManager.h>
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"
#include "IOPool/Streamer/interface/StreamDQMSerializer.h"
#include "TROOT.h"

using namespace edm;
using namespace std;
using namespace stor;
using boost::shared_ptr;

DQMServiceManager::DQMServiceManager(std::string filePrefix,
				     int  purgeTime,
				     int  readyTime,
                                     bool collateDQM,
				     bool archiveDQM,
				     bool useCompression,
				     int  compressionLevel):
  useCompression_(useCompression),
  compressionLevel_(compressionLevel),
  collateDQM_(collateDQM),
  archiveDQM_(archiveDQM),
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
  if(!collateDQM_) 
  {
    // no collation just pass the DQMEvent to the Event Server and return
    if (DQMeventServer_.get() != NULL) 
    {
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

    // At this point, purge old instances from the list
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
			object,
			msg.eventNumberAtUpdate());
      delete(object);
    }
  }
  
  // Now send the best DQMGroup for this grouping, which may 
  // not be the currently updated one (it may not yet be ready)
  DQMGroupDescriptor * descriptor = 
    getBestDQMGroupDescriptor(msg.topFolderName());
  if ( descriptor != NULL )
  {
    // Reserialize the data and give to DQM server
    DQMGroup    * group    = descriptor->group_;
    if ( !group->wasServedSinceUpdate() )
    {
      group->setServedSinceUpdate();
      DQMInstance * instance = descriptor->instance_;

      // Package list of TObjects into a DQMEvent::TObjectTable
      DQMEvent::TObjectTable table;

      int subFolderSize = 0;
      for ( std::map<std::string, DQMFolder *>::iterator i1 = 
	      group->dqmFolders_.begin(); i1 != group->dqmFolders_.end(); ++i1)
      {
	std::string folderName = i1->first;
	DQMFolder * folder = i1->second;
	for ( std::map<std::string, TObject *>::iterator i2 = 
	      folder->dqmObjects_.begin(); i2!=folder->dqmObjects_.end(); ++i2)
	{
	  std::string objectName = i2->first;
	  TObject *object = i2->second;
	  if ( object != NULL ) 
	  { 
	    if ( table.count(folderName) == 0 )
	    {
	      std::vector<TObject *> newObjectVector;
	      table[folderName] = newObjectVector;
	      subFolderSize += 2*sizeof(uint32) + folderName.length();
	    }
	    std::vector<TObject *> objectVector = table[folderName];
	    objectVector.push_back(object);
	  }
	}
      }

      edm::StreamDQMSerializer serializer;
      serializer.serializeDQMEvent(table,
				   useCompression_,
				   compressionLevel_);
      // Add space for header
      unsigned int sourceSize = serializer.currentSpaceUsed();
      unsigned int totalSize  = sourceSize 
	+ sizeof(DQMEventHeader)
	+ 12*sizeof(uint32)
	+ msg.releaseTag().length()
	+ msg.topFolderName().length()
	+ subFolderSize;
      unsigned char * buffer = (unsigned char *)malloc(totalSize);
      
      edm::Timestamp zeit( ( (unsigned long long)group->getLastUpdate()->GetSec() << 32 ) |
			   ( group->getLastUpdate()->GetNanoSec()));

      DQMEventMsgBuilder builder((void *)&buffer[0], 
				 totalSize,
				 instance->getRunNumber(),
				 group->getLastEvent(),
				 zeit,
				 instance->getLumiSection(),
				 instance->getInstance(),
				 msg.releaseTag(),
				 msg.topFolderName(),
				 table); 
      unsigned char * source = serializer.bufferPointer();
      std::copy(source,source+sourceSize, builder.eventAddress());
      builder.setEventLength(sourceSize);
      if ( useCompression_ ) { builder.setCompressionFlag(sourceSize); }
      DQMEventMsgView serveMessage(&buffer[0]);
      DQMeventServer_->processDQMEvent(msg);
      
      free(buffer);
    }
    delete(descriptor);
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
    if ( ( dqmInstances_[i]->getRunNumber()   == runNumber_ ) && 
	 ( dqmInstances_[i]->getLumiSection() == lumiSection_ ) && 
	 ( dqmInstances_[i]->getInstance()    == instance_ ) )
    { reply = dqmInstances_[i]; }
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
  std::vector<DQMInstance *>::iterator i0 = dqmInstances_.begin(); 
  while ( ( i0 != dqmInstances_.end() ) && ( n>minInstances) )
  {
    DQMInstance * instance = *i0;
    if ( instance->isStale(now.GetSec()) || writeAll)
    {
      if ( archiveDQM_ ) { instance->writeFile(filePrefix_);}
      delete(instance);
      reply++;
      i0 = dqmInstances_.erase(i0);
    }
    else
    {
      ++i0;
    }
    n = dqmInstances_.size();
  }
  return(reply);
}

DQMGroupDescriptor * DQMServiceManager::getBestDQMGroupDescriptor(std::string groupName)
{
  DQMGroupDescriptor * reply = NULL;
  TTimeStamp now;
  now.Set();

  DQMInstance * newestInstance = NULL;
  DQMGroup * newestGroup    = NULL;
  int maxTime = 0;
  for (std::vector<DQMInstance * >::iterator i0 = 
	 dqmInstances_.begin(); i0 != dqmInstances_.end() ; ++i0)
  {
    DQMInstance * instance = *i0;
    for (std::map<std::string, DQMGroup * >::iterator i1 = 
	   instance->dqmGroups_.begin(); 
	 i1 != instance->dqmGroups_.end() ; ++i1)
    {
      DQMGroup  * group     = i1->second;
      if ( group->isReady(now.GetSec()))
      {
        TTimeStamp * zeit = group->getLastUpdate();
        if ( ( zeit != NULL ) && ( zeit->GetSec() > maxTime ))
        {
	  maxTime = zeit->GetSec();
	  newestInstance  = instance; 
	  newestGroup     = group; 
        }
      }
    }
  }

  if ( ( newestInstance != NULL ) &&
       ( newestGroup    != NULL ) )
  { reply = new DQMGroupDescriptor(newestInstance,newestGroup); }

  return(reply);
}

