#ifndef Streamer_TestProducer_h
#define Streamer_TestProducer_h

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/InputSource.h"

#include <vector>
#include <memory>
#include <string>
#include <fstream>

namespace edmtestp
{
  class TestProducer : public edm::InputSource
  {
  public:
    typedef std::vector<char> RegBuffer;

    TestProducer(edm::ParameterSet const& pset,
		 edm::InputSourceDescription const& desc);
    virtual ~EventStreamInput();

    virtual std::auto_ptr<edm::EventPrincipal> read();

  private:  
    std::string filename_;
    std::ifstream ist_;
    RegBuffer regdata_;
    RegBuffer edata_;
    edm::EventDecoder event_dec_;
    edm::JobHeaderDecoder head_dec_;
  };
}

#endif
