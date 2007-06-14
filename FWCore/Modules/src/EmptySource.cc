/*----------------------------------------------------------------------
$Id: EmptySource.cc,v 1.4 2005/12/28 21:49:54 wmtan Exp $
----------------------------------------------------------------------*/

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
