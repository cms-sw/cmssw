#ifndef HLT_JOB_CNTLER_HPP
#define HLT_JOB_CNTLER_HPP

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "EventFilter/StorageManager/interface/FragmentCollector.h"
#include "EventFilter/StorageManager/interface/EPRunner.h"

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
    
    edm::EventBuffer& getFragmentQueue()
    { return collector_->getFragmentQueue(); }

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

