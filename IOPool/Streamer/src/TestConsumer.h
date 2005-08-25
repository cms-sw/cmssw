#ifndef IOP_TESTCONSUMER_H
#define IOP_TESTCONSUMER_H

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ProductRegistry.h"

namespace edmtest
{
  class Worker;

  class TestConsumer
  {
  public:
    TestConsumer(edm::ParameterSet const& ps, 
		 edm::ProductRegistry const& reg,
		 edm::EventBuffer* buf);

    ~TestConsumer();

    void bufferReady();
    void stop();
    void sendRegistry(void* buf, int len);

  private:
    Worker* worker_;
    edm::EventBuffer* bufs_;
  };
}

#endif

