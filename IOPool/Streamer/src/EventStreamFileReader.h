#ifndef STREAMER_EVENTSTREAMFILEREADER_H
#define STREAMER_EVENTSTREAMFILEREADER_H

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
  class EventStreamFileReader : public edm::InputSource
  {
  public:
    EventStreamFileReader(edm::ParameterSet const& pset,
		 edm::InputSourceDescription const& desc);
    virtual ~EventStreamFileReader();

    virtual std::auto_ptr<edm::EventPrincipal> read();

  private:  
    std::string filename_;
    std::ifstream ist_;
    edm::EventReader reader_;
  };

}

#endif

