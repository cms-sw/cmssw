/*----------------------------------------------------------------------
$Id: EmptySource.cc,v 1.3 2005/12/28 00:52:54 wmtan Exp $
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

  bool
  EmptySource::produce(edm::Event &) {
    return true;
  }
}
