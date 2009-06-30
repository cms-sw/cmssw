#ifndef _DQMSERVICEMANAGER_H_
#define _DQMSERVICEMANAGER_H_

// $Id: DQMServiceManager.h,v 1.4 2008/06/27 14:48:44 biery Exp $

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "xdata/UnsignedInteger32.h"
#include "xdata/Boolean.h"
#include "xdata/String.h"

#include "EventFilter/StorageManager/interface/DQMInstance.h"
#include "EventFilter/StorageManager/interface/DQMEventServer.h"
#include <IOPool/Streamer/interface/DQMEventMessage.h>
//#include <EventFilter/StorageManager/interface/DQMStreamService.h>

#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>

namespace stor
{
  class DQMServiceManager 
  {
    public:  
    
    explicit DQMServiceManager(std::string filePrefix = "/tmp/DQM",
                               int purgeTime = DEFAULT_PURGE_TIME,
                               int readyTime = DEFAULT_READY_TIME,
                               bool collateDQM = false,
                               bool archiveDQM = false,
                               int archiveInterval = 0,
                               bool useCompression = true,
                               int compressionLevel = 1);
     ~DQMServiceManager(); 
    
      void manageDQMEventMsg(DQMEventMsgView& msg);
      void stop();
      DQMGroupDescriptor * getBestDQMGroupDescriptor(std::string groupName);
      void setPurgeTime(int purgeTime) { purgeTime_ = purgeTime;}
      void setReadyTime(int readyTime) { readyTime_ = readyTime;}
      void setFilePrefix(std::string filePrefix)
      { filePrefix_ = filePrefix;}

      void setCollateDQM(bool collateDQM) { collateDQM_ = collateDQM; }
      void setArchiveDQM(bool archiveDQM) { archiveDQM_ = archiveDQM; }
      void setArchiveInterval(int archiveInterval) { archiveInterval_ = archiveInterval; }
      void setDQMEventServer(boost::shared_ptr<DQMEventServer>& es) 
      { DQMeventServer_ = es; }
      void setUseCompression(bool useCompression) 
      { useCompression_ = useCompression;}
      void setCompressionLevel(int compressionLevel) 
      { compressionLevel_ = compressionLevel;}

    protected:

      bool          useCompression_;
      int           compressionLevel_;
      bool          collateDQM_;
      bool          archiveDQM_;
      int           archiveInterval_;
      int           nUpdates_;
      std::string   filePrefix_;
      int           purgeTime_;
      int           readyTime_;
      int  writeAndPurgeDQMInstances(bool purgeAll=false);
      std::vector<DQMInstance *>    dqmInstances_;

      DQMInstance * findDQMInstance(int runNumber, 
                                    int lumiSection,
                                    int instance);

      boost::shared_ptr<DQMEventServer> DQMeventServer_;
      enum 
      {
        DEFAULT_PURGE_TIME = 120,
        DEFAULT_READY_TIME = 30
      }; 

      //      boost::shared_ptr<stor::DQMEventSelector>  dqmEventSelector_;
  };
}

#endif
