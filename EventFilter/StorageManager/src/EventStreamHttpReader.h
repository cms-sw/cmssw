#ifndef STREAMER_EVENTSTREAMHTTPREADER_H
#define STREAMER_EVENTSTREAMHTTPREADER_H

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/InputSource.h"

#include <vector>
#include <memory>
#include <string>
#include <fstream>

namespace edmtestp
{
  struct ReadData;

  class EventStreamHttpReader : public edm::InputSource
  {
  public:
    typedef std::vector<char> Buf;

    EventStreamHttpReader(edm::ParameterSet const& pset,
		 edm::InputSourceDescription const& desc);
    virtual ~EventStreamHttpReader();

    virtual std::auto_ptr<edm::EventPrincipal> read();
    virtual std::auto_ptr<edm::SendJobHeader> readHeader();

  private:  
    std::string sourceurl_;
    std::string eventurl_;
    std::string headerurl_;
    Buf buf_;
    edm::EventDecoder decoder_;
    edm::ProductRegistry prods_;
  };

}

#endif

