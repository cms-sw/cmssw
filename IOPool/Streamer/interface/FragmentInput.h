#ifndef Streamer_FragmentInput_h
#define Streamer_FragmentInput_h

// -*- C++ -*-

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include <memory>

namespace stor
{
  class FragmentInput : public edm::InputSource
  {
  public:
    FragmentInput(edm::ParameterSet const& pset,
		  edm::InputSourceDescription const& desc);
    virtual ~FragmentInput();

    virtual std::auto_ptr<edm::EventPrincipal> read();

  private:
    edm::EventExtractor extractor_;
  };
}

#endif
