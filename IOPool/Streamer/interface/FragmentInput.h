#ifndef HLT_FRAG_INPUT_H
#define HLT_FRAG_INPUT_H

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
