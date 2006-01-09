#ifndef HLT_EPRUNNER_H
#define HLT_EPRUNNER_H

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/ProblemTracker.h"
#include "IOPool/Streamer/interface/HLTInfo.h"

#include "boost/thread/thread.hpp"
#include "boost/shared_ptr.hpp"

#include <string>
#include <memory>

namespace stor
{
  class EPRunner
  {
  public:
    EPRunner(const std::string& config_string,
	     std::auto_ptr<HLTInfo>);
    ~EPRunner();

    void start();
    void join();

    const edm::ProductRegistry& getRegistry();
    HLTInfo* getInfo() { return info_; }

  private:
    EPRunner(const EPRunner&);
	EPRunner& operator=(const EPRunner&);

    static void run(EPRunner*);
    void dowork();
    HLTInfo* info_;
    edm::ServiceToken tok_;
	edm::AssertHandler ah_;
    edm::EventProcessor ep_;
    boost::shared_ptr<boost::thread> me_;
  };
}

#endif

