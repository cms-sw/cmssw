/*----------------------------------------------------------------------
$Id: VectorInputSource.cc,v 1.4 2006/04/06 23:26:29 wmtan Exp $
----------------------------------------------------------------------*/
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  VectorInputSource::VectorInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    EDInputSource(pset, desc) {}

  VectorInputSource::~VectorInputSource() {}

  void
  VectorInputSource::readMany(int number, EventPrincipalVector& result) {
    // Do we need any error handling (e.g. exception translation) here?
    this->readMany_(number, result);
  }
}
