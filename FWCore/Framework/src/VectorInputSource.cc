/*----------------------------------------------------------------------
$Id: VectorInputSource.cc,v 1.2 2006/01/18 23:26:22 wmtan Exp $
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/VectorInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  VectorInputSource::VectorInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    GenericInputSource(pset, desc) {}

  VectorInputSource::~VectorInputSource() {}

  void
  VectorInputSource::readMany(int number, EventPrincipalVector& result) {
    // Do we need any error handling (e.g. exception translation) here?
    this->readMany_(number, result);
  }
}
