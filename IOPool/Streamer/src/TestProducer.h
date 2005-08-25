#ifndef IOP_TESTPRODUCER_H
#define IOP_TESTPRODUCER_H

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ProductRegistry.h"

#include <vector>

namespace edmtestp
{
  class Worker;

  class TestProducer
  {
  public:
    typedef std::vector<char> RegBuffer;

    TestProducer(edm::ParameterSet const& ps, 
		 edm::ProductRegistry const& reg,
		 edm::EventBuffer* buf);

    ~TestProducer();

    void getRegistry(RegBuffer& copyhere);
    void stop();
    void needBuffer();

  private:
    Worker* worker_;
    edm::EventBuffer* bufs_;
  };
}

#endif
