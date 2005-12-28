/*----------------------------------------------------------------------
$Id: EmptySource.cc,v 1.2 2005/11/14 21:22:31 wmtan Exp $
----------------------------------------------------------------------*/

#include <stdexcept>
#include <memory>


#include "FWCore/Modules/src/EmptySource.h"

namespace edm {
  EmptySource::EmptySource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    GeneratedInputSource(pset, desc)
  { }

  EmptySource::~EmptySource() {
  }

  void
  EmptySource::produce(edm::Event &) {
  }
}
