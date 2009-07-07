#ifndef HLT_JOB_CNTLER_HPP
#define HLT_JOB_CNTLER_HPP
// $Id: JobController.h,v 1.21.2.4 2008/11/15 20:01:58 biery Exp $

#include "EventFilter/StorageManager/interface/FragmentCollector.h"
#include "EventFilter/StorageManager/interface/EventServer.h"
#include "EventFilter/StorageManager/interface/DQMEventServer.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"
#include "EventFilter/StorageManager/interface/SMFUSenderList.h"

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/Messages.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include "log4cplus/logger.h"

#include <string>

namespace stor
{

  class JobController
  {
  public:
    JobController(const std::string& my_config,
		  log4cplus::Logger& applicationLogger,
		  FragmentCollector::Deleter);

    ~JobController();

    void start();
    void stop();
    void join();

    void receiveMessage(FragEntry& entry);

    edm::EventBuffer& getFragmentQueue()
    { return collector_->getFragmentQueue(); }

    void setEventServer(boost::shared_ptr<EventServer>& es)
    {
      if (collector_.get() != NULL) collector_->setEventServer(es);
      eventServer_ = es;
    }
    boost::shared_ptr<EventServer>& getEventServer() { return eventServer_; }

    void setDQMEventServer(boost::shared_ptr<DQMEventServer>& es)
    {
      if (collector_.get() != NULL) collector_->setDQMEventServer(es);
      DQMeventServer_ = es;
    }
    boost::shared_ptr<DQMEventServer>& getDQMEventServer() { return DQMeventServer_; }

    void setInitMsgCollection(boost::shared_ptr<InitMsgCollection>& imColl)
    {
      if (collector_.get() != NULL) collector_->setInitMsgCollection(imColl);
      initMsgCollection_ = imColl;
    }
    boost::shared_ptr<InitMsgCollection>& getInitMsgCollection() { return initMsgCollection_; }

    void setSMRBSenderList(SMFUSenderList* senderList) {
      if (collector_.get() != NULL) collector_->setSMRBSenderList(senderList);
      smRBSenderList_ = senderList;
    }

    void setNumberOfFileSystems(int disks)    { collector_->setNumberOfFileSystems(disks); }
    void setFileCatalog(std::string catalog)  { collector_->setFileCatalog(catalog); }
    void setSourceId(std::string sourceId)    { collector_->setSourceId(sourceId); }
    void setCollateDQM(bool collateDQM)       { collector_->setCollateDQM(collateDQM);}
    void setArchiveDQM(bool archiveDQM)       { collector_->setArchiveDQM(archiveDQM);}
    void setArchiveIntervalDQM(int archiveInterval) {
      collector_->setArchiveIntervalDQM(archiveInterval);
    }
    void setPurgeTimeDQM(int purgeTimeDQM)    { collector_->setPurgeTimeDQM(purgeTimeDQM);}
    void setReadyTimeDQM(int readyTimeDQM)    { collector_->setReadyTimeDQM(readyTimeDQM);}
    void setFilePrefixDQM(std::string filePrefixDQM)  { collector_->setFilePrefixDQM(filePrefixDQM);}
    void setUseCompressionDQM(bool useCompressionDQM)
    { collector_->setUseCompressionDQM(useCompressionDQM);}
    void setCompressionLevelDQM(bool compressionLevelDQM)
    { collector_->setCompressionLevelDQM(compressionLevelDQM);}
    void setFileClosingTestInterval(int value) { fileClosingTestInterval_ = value; }

    std::list<std::string>& get_filelist() { return collector_->get_filelist(); }
    std::list<std::string>& get_currfiles() { return collector_->get_currfiles(); }
    std::vector<uint32>& get_storedEvents() { return collector_->get_storedEvents(); }
    std::vector<std::string>& get_storedNames() { return collector_->get_storedNames(); }
    boost::shared_ptr<stor::SMOnlyStats> get_stats() { return collector_->get_stats(); }

  private:
    void init(const std::string& my_config,FragmentCollector::Deleter);
    void processCommands();
    static void run(JobController*);

    boost::shared_ptr<FragmentCollector> collector_;
    boost::shared_ptr<EventServer> eventServer_;
    boost::shared_ptr<DQMEventServer> DQMeventServer_;
    boost::shared_ptr<InitMsgCollection> initMsgCollection_;
    SMFUSenderList* smRBSenderList_;

    int fileClosingTestInterval_;
    log4cplus::Logger& applicationLogger_;

    boost::shared_ptr<boost::thread> me_;
  };
}

#endif

