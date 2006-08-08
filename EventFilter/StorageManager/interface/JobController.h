#ifndef HLT_JOB_CNTLER_HPP
#define HLT_JOB_CNTLER_HPP

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "EventFilter/StorageManager/interface/FragmentCollector.h"
#include "EventFilter/StorageManager/interface/EPRunner.h"
#include "IOPool/Streamer/interface/EventMessage.h"

// added by HWKC for Event Server
#include "IOPool/Streamer/interface/Messages.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include <string>

namespace stor
{

  class JobController
  {
  public:
    JobController(const std::string& fu_config,
		  const std::string& my_config,
		  FragmentCollector::Deleter);
    JobController(const edm::ProductRegistry& reg,
		  const std::string& my_config,
		  FragmentCollector::Deleter);

    ~JobController();

    void start();
    void stop();
    void join();

    void receiveMessage(FragEntry& entry);

    const edm::ProductRegistry& products() const { return prods_; }
    edm::ProductRegistry& products() { return prods_; }

    const edm::ProductRegistry& smproducts() const { return ep_runner_->getRegistry(); }
    
    edm::EventBuffer& getFragmentQueue()
    { return collector_->getFragmentQueue(); }

    // added for Event Server by HWKC so SM can get event from ring buffer
    bool isEmpty() { return collector_->esbuf_isEmpty(); }
    bool isFull() { return collector_->esbuf_isFull(); }
//HEREHERE
    EventMsgView pop_front() {return collector_->esbuf_pop_front();}
    void push_back(EventMsgView msg) 
      { collector_->esbuf_push_back(msg); }
//HEREHERE
    void set_oneinN(int N) { collector_->set_esbuf_oneinN(N); }
    edm::ServiceToken getToken() { return ep_runner_->getToken();}

//HEREHEREHERE
    // a hack to get hlt and l1 counts into fragColl
    void set_hlt_bit_count(uint32 count) { collector_->set_hlt_bit_count(count);}
    void set_l1_bit_count(uint32 count) { collector_->set_l1_bit_count(count);}
    void set_outoption(bool stream_only) { collector_->set_outoption(stream_only);}
    void set_outfile(std::string outfilestart, unsigned long maxFileSize,
           double highWaterMark, std::string path, std::string mpath) 
           { collector_->set_outfile(outfilestart,maxFileSize,highWaterMark,path,mpath);}
    std::list<std::string> get_filelist() { return collector_->get_filelist(); }
    std::string get_currfile() { return collector_->get_currfile(); }

  private:
    void init(const std::string& my_config,FragmentCollector::Deleter);
    void setRegistry(const std::string& fu_config);
    void processCommands();
    static void run(JobController*);
    edm::ProductRegistry prods_;

    boost::shared_ptr<FragmentCollector> collector_;
    boost::shared_ptr<EPRunner> ep_runner_;

    boost::shared_ptr<boost::thread> me_;
  };
}

#endif

