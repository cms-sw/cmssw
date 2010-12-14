//
// File: EventFilter/StorageManager/src/DQMServiceManager.cc
//
// (W.Badgett)
//
// $Id: DQMServiceManager.cc,v 1.14 2010/05/17 15:59:10 mommsen Exp $
//
// Note: this class is no longer used in the StorageManager, but is still
// required by the SMProxyServer (Remi Mommsen, May 5, 2009)
//

#include "FWCore/Utilities/interface/DebugMacros.h"
#include <EventFilter/StorageManager/interface/DQMServiceManager.h>
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"
#include "IOPool/Streamer/interface/StreamDQMSerializer.h"
#include "TROOT.h"

#include <limits>
#include <unistd.h>

using namespace edm;
using namespace std;
using namespace stor;
using boost::shared_ptr;

DQMServiceManager::DQMServiceManager(std::string filePrefix,
				     int  purgeTime,
				     int  readyTime,
                                     bool collateDQM,
				     bool archiveDQM,
				     int archiveInterval,
				     bool useCompression,
                                     int  compressionLevel,
                                     int expectedUpdates):
  useCompression_(useCompression),
  compressionLevel_(compressionLevel),
  collateDQM_(collateDQM),
  archiveDQM_(archiveDQM),
  archiveInterval_(archiveInterval),
  nUpdates_(0),
  sentEvents_(0),
  filePrefix_(filePrefix),
  purgeTime_(purgeTime),
  readyTime_(readyTime),
  expectedUpdates_(expectedUpdates)
{
  dqmInstances_.reserve(20);

  gROOT->SetBatch(kTRUE);
  int got_host = gethostname(host_name_, 255);
  if(got_host != 0) strcpy(host_name_, "noHostNameFoundOrTooLong");
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

  DQMInstance * dqm = findDQMInstance(msg.runNumber(),
				      msg.lumiSection(),
				      msg.updateNumber());
  if ( dqm == NULL ) 
  {
    dqm = new DQMInstance(msg.runNumber(),
			  msg.lumiSection(),
			  msg.updateNumber(),
			  purgeTime_,
                          readyTime_,
                          expectedUpdates_);
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

  dqm->addEvent(msg.topFolderName(), toTablePtr);
  
  // Now send the best DQMGroup for this grouping, which may 
  // not be the currently updated one (it may not yet be ready)
  DQMGroupDescriptor * descriptor = 
    getBestDQMGroupDescriptor(msg.topFolderName());
  if ( descriptor != NULL )
  {
    // Reserialize the data and give to DQM server
    DQMGroup * group = descriptor->group_;
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
	      subFolderSize += 2*sizeof(uint32_t) + folderName.length();
	    }
	    std::vector<TObject *> &objectVector = table[folderName];
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
	+ 12*sizeof(uint32_t)
	+ msg.releaseTag().length()
	+ msg.topFolderName().length()
	+ subFolderSize;
      unsigned char * buffer = (unsigned char *)malloc(totalSize);
      
      edm::Timestamp zeit( ( (unsigned long long)group->getLastUpdate()->GetSec() << 32 ) |
			   ( group->getLastUpdate()->GetNanoSec()));

      DQMEventMsgBuilder builder((void *)&buffer[0], 
				 totalSize,
				 instance->getRunNumber(),
                                 ++sentEvents_, //group->getLastEvent(),
				 zeit,
				 instance->getLumiSection(),
				 instance->getUpdateNumber(),
                                 (uint32_t)serializer.adler32_chksum(),
                                 host_name_,
				 msg.releaseTag(),
				 msg.topFolderName(),
				 table); 
      unsigned char * source = serializer.bufferPointer();
      std::copy(source,source+sourceSize, builder.eventAddress());
      builder.setEventLength(sourceSize);
      if ( useCompression_ ) { builder.setCompressionFlag(serializer.currentEventSize()); }
      DQMEventMsgView serveMessage(&buffer[0]);
      DQMeventServer_->processDQMEvent(serveMessage);
      
      free(buffer);
    }
    delete(descriptor);
  }
}

void DQMServiceManager::setParameters(DQMProcessingParams const& dqmParams)
{
  useCompression_ = dqmParams._useCompressionDQM;
  compressionLevel_ = dqmParams._compressionLevelDQM;
  collateDQM_ = dqmParams._collateDQM;
  archiveDQM_ = dqmParams._archiveDQM;
  archiveInterval_ = dqmParams._archiveIntervalDQM;
  filePrefix_ = dqmParams._filePrefixDQM;
  purgeTime_ = dqmParams._purgeTimeDQM.total_seconds();
  readyTime_ = dqmParams._readyTimeDQM.total_seconds();
}

DQMInstance * DQMServiceManager::findDQMInstance(int runNumber, 
						 int lumiSection, 
						 int instance)
{
  DQMInstance * reply = NULL;
  int n = dqmInstances_.size();
  for ( int i=0; (i<n) && (reply==NULL); i++)
  {
    if ( ( dqmInstances_[i]->getRunNumber()   == runNumber ) && 
	 ( dqmInstances_[i]->getLumiSection() == lumiSection ) && 
	 ( dqmInstances_[i]->getUpdateNumber()== instance ) )
    { reply = dqmInstances_[i]; }
  }
  return(reply);
}

int DQMServiceManager::writeAndPurgeDQMInstances(bool writeAll)
{
  int reply = 0;

  TTimeStamp now;
  now.Set();

  // Always keep at least one instance in memory, unless we're
  // writing everything
  int minInstances = 1;
  if ( writeAll ) { minInstances=0;}

  // if we're writing everything, find the last instance that is ready
  int listSizeWithOneReady = 0;
  if ( writeAll )
  {
    std::vector<DQMInstance *>::reverse_iterator r0 = dqmInstances_.rbegin();
    while ( ( r0 != dqmInstances_.rend() ) )
    {
      ++listSizeWithOneReady;
      DQMInstance * instance = *r0;
      if (instance->isReady())
      {
        break;
      }
      ++r0;
    }
  }

  int n = dqmInstances_.size();
  std::vector<DQMInstance *>::iterator i0 = dqmInstances_.begin(); 
  while ( ( i0 != dqmInstances_.end() ) && ( n>minInstances) )
  {
    DQMInstance * instance = *i0;
    if ( instance->isStale(now.GetSec()) || writeAll)
    {
      if (archiveDQM_ && instance->isReady() &&
          ((archiveInterval_ > 0 &&
            (instance->getLumiSection() % archiveInterval_) == 0)
           || (writeAll && n == listSizeWithOneReady)))
      {
        instance->writeFile(filePrefix_,
                            (writeAll && n == listSizeWithOneReady));
      }
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
    DQMGroup * group = instance->getDQMGroup(groupName);
    if (group != NULL && group->isReady(now.GetSec()))
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

  if ( ( newestInstance != NULL ) &&
       ( newestGroup    != NULL ) )
  { reply = new DQMGroupDescriptor(newestInstance,newestGroup); }
  return(reply);
}

