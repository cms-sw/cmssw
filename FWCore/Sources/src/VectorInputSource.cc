/*----------------------------------------------------------------------
$Id: VectorInputSource.cc,v 1.2 2007/06/14 21:03:40 wmtan Exp $
----------------------------------------------------------------------*/
#include "FWCore/Sources/interface/VectorInputSource.h"

namespace edm {

  VectorInputSource::VectorInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    EDInputSource(pset, desc) {}

  VectorInputSource::~VectorInputSource() {}

  void
  VectorInputSource::readMany(int number, EventPrincipalVector& result) {
    this->readMany_(number, result);
  }

  void
  VectorInputSource::readMany(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber) {
    this->readMany_(number, result, id, fileSeqNumber);
  }

  void
  VectorInputSource::readManyRandom(int number, EventPrincipalVector& result) {
    this->readManyRandom_(number, result);
  }
}
