/*----------------------------------------------------------------------
$Id: DummySource.cc,v 1.6 2007/04/13 19:12:38 wmtan Exp $
----------------------------------------------------------------------*/

#include <stdexcept>
#include <memory>

#include "FWCore/Framework/test/DummySource.h"

namespace edm {
  DummySource::DummySource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    GeneratedInputSource(pset, desc)
  { }

  DummySource::~DummySource() {
  }

  bool
  DummySource::produce(edm::Event &) {
    return true;
  }
}
