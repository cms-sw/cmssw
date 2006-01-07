/*----------------------------------------------------------------------
$Id: VectorInputSource.cc,v 1.0 2006/01/06 00:30:39 wmtan Exp $
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/VectorInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  VectorInputSource::VectorInputSource(InputSourceDescription const& desc) :
    InputSource(desc) {}

  VectorInputSource::~VectorInputSource() {}

  void
  VectorInputSource::readMany(int number, std::vector<EventPrincipal*>& result) {
    // Do we need any error handling (e.g. exception translation) here?
    this->readMany_(number, result);
  }
}
