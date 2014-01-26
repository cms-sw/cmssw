#ifndef EVF_FASTMONITORINGTHREAD
#define EVF_FASTMONITORINGTHREAD

#include "EventFilter/Utilities/interface/FastMonitor.h"

#include "boost/thread/thread.hpp"

#include <iostream>
#include <vector>



using namespace jsoncollector;

namespace evf{

  class FastMonitoringService;

  class FastMonitoringThread{
  public:
    // a copy of the Framework/EventProcessor states 
    enum Macrostate { sInit = 0, sJobReady, sRunGiven, sRunning, sStopping,
		      sShuttingDown, sDone, sJobEnded, sError, sErrorEnded, sEnd, sInvalid,MCOUNT}; 
    struct StreamMonitorData
    {
      //global state
      Macrostate macrostate_ = FastMonitoringThread::sInit;
      IntJ macrostateJ_;

      //per lumi global (we get this from elsewhere..
      //unsigned int lumisection_ = 0; //only updated on beginLumi signal
      // accummulated size of processed files over a lumi (in Bytes)

      // Throughput, MB/s
      DoubleJ throughputJ_;
      // Average time to obtain a file to read (ms)
      DoubleJ avgLeadTimeJ_;
      // Number of files processed during lumi section
      std::map<unsigned int,IntJ> filesProcessedDuringLumi_;

      //stream

      // Micro, mini, macrostate numbers
      std::vector<const void*> ministate_;
      std::vector<IntJ> ministateJ_; //stream

      std::vector<const void*> microstate_;
      std::vector<IntJ> microstateJ_; //stream

      // Processed events count
      std::vector<unsigned int> processed_;
      std::vector<IntJ> processedJ_;

      std::vector<unsigned int> streamlumi_ = 0; //maybe will need this for updates...?
      //std::vector<unsigned int> eventnumber_;
      //unsigned int prescaleindex_; // ditto

      //monitored for JSON

      boost::shared_ptr<FastMonitor> jsonMonitor_;//collector

    };

    //constructor
    FastMonitoringThread() : m_stoprequest(false){

      //set up monitorables here

      m_data.macrostateJ_.setName("Macrostate");
      m_data.ministateJ_.setName("Ministate");
      m_data.microstateJ_.setName("Microstate");
      m_data.processedJ_.setName("Processed");
      m_data.throughputJ_.setName("Throughput");
      m_data.avgLeadTimeJ_.setName("AverageLeadTime");
      m_data.filesProcessedDuringLumi_.setName("FilesProcessed");

      monParams.push_back(&m_data.macrostateJ_);

      monParams.push_back(&m_data.ministateJ_);
      monParams.push_back(&m_data.microstateJ_);
      monParams.push_back(&m_data.processedJ_);

      monParams.push_back(&m_data.throughputJ_);
      monParams.push_back(&m_data.avgLeadTimeJ_);
      monParams.push_back(&m_data.filesProcessedDuringLumi_);


      m_data.macrostateJ_.setName("Macrostate");
      m_data.ministateJ_.setName("Ministate");
      m_data.microstateJ_.setName("Microstate");
      m_data.processedJ_.setName("Processed");
      m_data.throughputJ_.setName("Throughput");
      m_data.avgLeadTimeJ_.setName("AverageLeadTime");
      m_data.filesProcessedDuringLumi_.setName("FilesProcessed");

    }

    void resetFastMonitor(std::string const& microStateDefPath) {
      m_data.jsonMonitor_.reset(new FastMonitor(monParams,microStateDefPath));
    }

    void start(void (FastMonitoringService::*fp)(),FastMonitoringService *cp){
      assert(!m_thread);
      m_thread = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(fp,cp)));
    }
    void stop(){
      assert(m_thread);
      m_stoprequest=true;
      m_thread->join();
    }

  private:

    volatile bool m_stoprequest;
    boost::shared_ptr<boost::thread> m_thread;
    std::vector<MonitorData> m_data;
    boost::mutex lock_;
    boost::mutex monlock_;

    std::vector<JsonMonitorable*> monParams;

    friend class FastMonitoringService;
  };
} //end namespace evf
#endif
