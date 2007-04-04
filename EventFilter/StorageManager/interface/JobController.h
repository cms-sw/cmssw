#ifndef HLT_JOB_CNTLER_HPP
#define HLT_JOB_CNTLER_HPP
// $Id: JobController.h,v 1.14 2006/12/22 09:48:17 klute Exp $

#include "EventFilter/StorageManager/interface/FragmentCollector.h"
#include "EventFilter/StorageManager/interface/EventServer.h"
#include "EventFilter/StorageManager/interface/DQMEventServer.h"

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/Messages.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include <string>

namespace stor
{

  class JobController
  {
  public:
    // remove next ctor later
    JobController(const std::string& fu_config,
		  const std::string& my_config,
		  FragmentCollector::Deleter);
    JobController(const std::string& my_config,
		  FragmentCollector::Deleter);
    JobController(const edm::ProductRegistry& reg,
		  const std::string& my_config,
		  FragmentCollector::Deleter);

    ~JobController();

    void start();
    void stop();
    void join();

    void receiveMessage(FragEntry& entry);

    edm::EventBuffer& getFragmentQueue()
    { return collector_->getFragmentQueue(); }

    bool isEmpty() { return collector_->esbuf_isEmpty(); }
    bool isFull() { return collector_->esbuf_isFull(); }
    EventMsgView pop_front() {return collector_->esbuf_pop_front();}
    void push_back(EventMsgView msg) 
      { collector_->esbuf_push_back(msg); }

    void set_oneinN(int N) { collector_->set_esbuf_oneinN(N); }
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

    void set_outoption(bool stream_only)      { collector_->set_outoption(stream_only);}
    void setNumberOfFileSystems(int disks)    { collector_->setNumberOfFileSystems(disks); }
    void setFileCatalog(std::string catalog)  { collector_->setFileCatalog(catalog); }
    void setSourceId(std::string sourceId)    { collector_->setSourceId(sourceId); }

    std::list<std::string>& get_filelist() { return collector_->get_filelist(); }
    std::list<std::string>& get_currfiles() { return collector_->get_currfiles(); }

  private:
    void init(const std::string& my_config,FragmentCollector::Deleter);
    void processCommands();
    static void run(JobController*);

    boost::shared_ptr<FragmentCollector> collector_;
    boost::shared_ptr<EventServer> eventServer_;
    boost::shared_ptr<DQMEventServer> DQMeventServer_;

    boost::shared_ptr<boost::thread> me_;
  };
}

#endif

