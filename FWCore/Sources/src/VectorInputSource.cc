/*----------------------------------------------------------------------
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
  VectorInputSource::readManyRandom(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    this->readManyRandom_(number, result, fileSeqNumber);
  }

  void
  VectorInputSource::readManySequential(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    this->readManySequential_(number, result, fileSeqNumber);
  }

  void
  VectorInputSource::dropUnwantedBranches(std::vector<std::string> const& wantedBranches) {
    this->dropUnwantedBranches_(wantedBranches);
  }
}
