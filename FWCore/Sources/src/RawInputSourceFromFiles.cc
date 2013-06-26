#include "FWCore/Sources/interface/RawInputSourceFromFiles.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  RawInputSourceFromFiles::RawInputSourceFromFiles(ParameterSet const& pset, InputSourceDescription const& desc) :
    RawInputSource(pset, desc),
    FromFiles(pset) {
  }

  RawInputSourceFromFiles::~RawInputSourceFromFiles() {}

  void
  RawInputSourceFromFiles::fillDescription(ParameterSetDescription & desc) {
    RawInputSource::fillDescription(desc);
    FromFiles::fillDescription(desc);
  }
}

