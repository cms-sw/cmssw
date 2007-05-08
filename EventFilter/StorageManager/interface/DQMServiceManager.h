#ifndef _DQMSERVICEMANAGER_H_
#define _DQMSERVICEMANAGER_H_

// $Id$

#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Framework/interface/EventSelector.h"

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "xdata/UnsignedInteger32.h"
#include "xdata/Boolean.h"
#include "xdata/String.h"

#include "EventFilter/StorageManager/interface/DQMInstance.h"
#include "EventFilter/StorageManager/interface/DQMEventServer.h"
#include <IOPool/Streamer/interface/DQMEventMessage.h>
//#include <EventFilter/StorageManager/interface/DQMStreamService.h>

#include "TApplication.h"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <list>
#include <string>

namespace stor
{
  class DQMServiceManager 
  {
    public:  
    
    explicit DQMServiceManager(std::string filePrefix = "/tmp/DQM",
			       int purgeTime = DEFAULT_PURGE_TIME,
			       int readyTime = DEFAULT_READY_TIME,
                               bool collateDQM = true);
     ~DQMServiceManager(); 
    
      void manageDQMEventMsg(DQMEventMsgView& msg);
      void stop();
      DQMInstance * getLastDQMInstance();

      void setPurgeTime(int purgeTime) { purgeTime_ = purgeTime;}
      void setReadyTime(int readyTime) { readyTime_ = readyTime;}
      void setFilePrefix(std::string filePrefix)
      { filePrefix_ = filePrefix;}

      void setCollateDQM(bool collateDQM) { collateDQM_ = collateDQM; }
      void setDQMEventServer(boost::shared_ptr<DQMEventServer>& es) { DQMeventServer_ = es; }

    protected:

      bool          collateDQM_;
      int           runNumber_;
      int           lumiSection_;
      int           instance_;
      int           nUpdates_;
      std::string   filePrefix_;
      int           purgeTime_;
      int           readyTime_;
      TApplication *rootApplication_;
      int  writeAndPurgeDQMInstances(bool purgeAll=false);
      std::vector<DQMInstance *>    dqmInstances_;

      DQMInstance * findDQMInstance(int runNumber_, 
				    int lumiSection_,
				    int instance_);

      boost::shared_ptr<DQMEventServer> DQMeventServer_;
      enum 
      {
	DEFAULT_PURGE_TIME = 20,
	DEFAULT_READY_TIME = 10
      }; 

      //      boost::shared_ptr<stor::DQMEventSelector>  dqmEventSelector_;
  };
}

#endif
