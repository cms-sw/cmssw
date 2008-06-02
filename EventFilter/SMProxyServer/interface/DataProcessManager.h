#ifndef SMPS_DATA_PROCESS_MANAGER_HPP
#define SMPS_DATA_PROCESS_MANAGER_HPP
// $Id: DataProcessManager.h,v 1.5 2008/02/11 14:54:52 biery Exp $

#include "EventFilter/StorageManager/interface/EventServer.h"
#include "EventFilter/StorageManager/interface/DQMEventServer.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/DQMServiceManager.h"
#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"
#include "EventFilter/StorageManager/interface/ForeverCounter.h"
#include "EventFilter/StorageManager/interface/RollingIntervalCounter.h"
#include "FWCore/Utilities/interface/CPUTimer.h"

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/Messages.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include <sys/time.h>
#include <string>
#include <vector>

namespace stor
{

  class DataProcessManager
  {
  public:
    enum STATS_TIME_FRAME { SHORT_TERM = 0, LONG_TERM = 1 };
    enum STATS_TIMING_TYPE { EVENT_FETCH = 10, DQMEVENT_FETCH = 11 };

    typedef std::vector<char> Buf;
    typedef std::map<std::string, unsigned int> RegConsumer_map;
    typedef std::map<std::string, bool> HeaderConsumer_map;
    typedef std::map<std::string, struct timeval> LastReqTime_map;

    DataProcessManager();

    ~DataProcessManager();

    void start();
    void stop();
    void join();

    void setEventServer(boost::shared_ptr<EventServer>& es)
    {
      eventServer_ = es;
    }
    boost::shared_ptr<EventServer>& getEventServer() { return eventServer_; }

    void setInitMsgCollection(boost::shared_ptr<InitMsgCollection>& imColl)
    {
      initMsgCollection_ = imColl;
    }
    boost::shared_ptr<InitMsgCollection>& getInitMsgCollection() { return initMsgCollection_; }

    void setMaxEventRequestRate(double rate);
    void setMaxDQMEventRequestRate(double rate);

    void setCollateDQM(bool collateDQM)
    { dqmServiceManager_->setCollateDQM(collateDQM); }

    void setArchiveDQM(bool archiveDQM)
    { dqmServiceManager_->setArchiveDQM(archiveDQM); }

    void setPurgeTimeDQM(int purgeTimeDQM)
    { dqmServiceManager_->setPurgeTime(purgeTimeDQM);}

    void setReadyTimeDQM(int readyTimeDQM)
    { dqmServiceManager_->setReadyTime(readyTimeDQM);}

    void setFilePrefixDQM(std::string filePrefixDQM)
    { dqmServiceManager_->setFilePrefix(filePrefixDQM);}

    void setUseCompressionDQM(bool useCompressionDQM)
    { dqmServiceManager_->setUseCompression(useCompressionDQM);}

    void setCompressionLevelDQM(int compressionLevelDQM)
    { dqmServiceManager_->setCompressionLevel(compressionLevelDQM);}

    void setDQMEventServer(boost::shared_ptr<DQMEventServer>& es)
    {
      // The auto_ptr still owns the memory after this get()
      if (dqmServiceManager_.get() != NULL) dqmServiceManager_->setDQMEventServer(es);
      DQMeventServer_ = es;
    }
    boost::shared_ptr<DQMEventServer>& getDQMEventServer() { return DQMeventServer_; }
    boost::shared_ptr<stor::DQMServiceManager>& getDQMServiceManager() { return dqmServiceManager_; }

    edm::EventBuffer& getCommandQueue() { return *cmd_q_; }

    void setConsumerName(std::string s) { consumerName_ = s; }
    void setDQMConsumerName(std::string s) { DQMconsumerName_ = s; }

    void addSM2Register(std::string smURL);
    void addDQMSM2Register(std::string DQMsmURL);
    bool haveRegWithDQMServer();
    bool haveRegWithEventServer();
    bool haveHeader();

    // *** for performance measurements
    unsigned long receivedevents() { return receivedEvents_; }
    unsigned long receivedDQMevents() { return receivedDQMEvents_; }
    double bandwidth() { return pmeter_->bandwidth(); }
    double rate() { return pmeter_->rate(); }
    double latency() { return pmeter_->latency(); }
    double meanbandwidth() { return pmeter_->meanbandwidth(); }
    double maxbandwidth() { return pmeter_->maxbandwidth(); }
    double minbandwidth() { return pmeter_->minbandwidth(); }
    double meanrate() { return pmeter_->meanrate(); }
    double meanlatency() { return pmeter_->meanlatency(); }
    unsigned long totalsamples() { return pmeter_->totalsamples(); }
    double totalvolumemb() { return pmeter_->totalvolumemb(); }
    double duration() { return pmeter_->duration(); }
    stor::SMPerfStats getStats() { return stats_; }
    void addMeasurement(unsigned long size);
    void setSamples(unsigned long num_samples) { pmeter_->setSamples(num_samples); }
    unsigned long samples() { return pmeter_->samples(); }

    double getSampleCount(STATS_TIME_FRAME timeFrame = SHORT_TERM,
                          STATS_TIMING_TYPE timingType = EVENT_FETCH,
                          double currentTime = BaseCounter::getCurrentTime());
    double getAverageValue(STATS_TIME_FRAME timeFrame = SHORT_TERM,
                           STATS_TIMING_TYPE timingType = EVENT_FETCH,
                           double currentTime = BaseCounter::getCurrentTime());
    double getDuration(STATS_TIME_FRAME timeFrame = SHORT_TERM,
                       STATS_TIMING_TYPE timingType = EVENT_FETCH,
                       double currentTime = BaseCounter::getCurrentTime());

  private:
    void init();
    void processCommands();
    static void run(DataProcessManager*);
    void getEventFromAllSM();
    double getTime2Wait(std::string smURL);
    void setTime2Now(std::string smURL);
    bool getOneEventFromSM(std::string smURL, double& time2wait);
    void getDQMEventFromAllSM();
    double getDQMTime2Wait(std::string smURL);
    void setDQMTime2Now(std::string smURL);
    bool getOneDQMEventFromSM(std::string smURL, double& time2wait);

    bool registerWithAllSM();
    bool registerWithAllDQMSM();
    int registerWithSM(std::string smURL);
    int registerWithDQMSM(std::string smURL);
    bool getAnyHeaderFromSM();
    bool getHeaderFromAllSM();
    bool getHeaderFromSM(std::string smURL);
    void waitBetweenRegTrys();

    edm::EventBuffer* cmd_q_;

    bool alreadyRegistered_;
    bool alreadyRegisteredDQM_;
    bool headerRefetchRequested_;
    std::vector<unsigned char> buf_;

    std::vector<std::string> smList_;
    RegConsumer_map smRegMap_;
    HeaderConsumer_map smHeaderMap_;
    std::vector<std::string> DQMsmList_;
    RegConsumer_map DQMsmRegMap_;
    std::string eventpage_;
    std::string regpage_;
    std::string DQMeventpage_;
    std::string DQMregpage_;
    std::string headerpage_;
    char subscriptionurl_[2048];

    std::string consumerName_;
    std::string consumerPriority_;
    std::string consumerPSetString_;
    int headerRetryInterval_; // seconds
    double minEventRequestInterval_;
    unsigned int consumerId_;
    LastReqTime_map lastReqMap_;
    double minDQMEventRequestInterval_;
    unsigned int DQMconsumerId_;
    LastReqTime_map lastDQMReqMap_;
    std::string DQMconsumerName_;
    std::string DQMconsumerPriority_;
    std::string consumerTopFolderName_;

    //std::auto_ptr<stor::DQMServiceManager> dqmServiceManager_;
    boost::shared_ptr<stor::DQMServiceManager> dqmServiceManager_;

    boost::shared_ptr<EventServer> eventServer_;
    boost::shared_ptr<DQMEventServer> DQMeventServer_;
    boost::shared_ptr<InitMsgCollection> initMsgCollection_;

    boost::shared_ptr<boost::thread> me_;

    // *** for performance measurements
    unsigned long receivedEvents_;
    unsigned long receivedDQMEvents_;
    stor::SMPerformanceMeter *pmeter_;
    // *** measurements for last set of samples
    xdata::UnsignedInteger32 samples_; // number of samples per measurement
    stor::SMPerfStats stats_;

    // statistics (long term and short term)
    edm::CPUTimer eventFetchTimer_;
    edm::CPUTimer dqmFetchTimer_;
    boost::shared_ptr<ForeverCounter> ltEventFetchTimeCounter_;
    boost::shared_ptr<RollingIntervalCounter> stEventFetchTimeCounter_;
    boost::shared_ptr<ForeverCounter> ltDQMFetchTimeCounter_;
    boost::shared_ptr<RollingIntervalCounter> stDQMFetchTimeCounter_;
  };
}

#endif

