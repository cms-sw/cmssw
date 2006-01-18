/*----------------------------------------------------------------------
$Id: VectorInputSource.cc,v 1.1 2006/01/07 20:41:12 wmtan Exp $
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/VectorInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  VectorInputSource::VectorInputSource(InputSourceDescription const& desc) :
    InputSource(desc) {}

  VectorInputSource::~VectorInputSource() {}

  void
  VectorInputSource::readMany(int number, EventPrincipalVector& result) {
    // Do we need any error handling (e.g. exception translation) here?
    this->readMany_(number, result);
  }
}
