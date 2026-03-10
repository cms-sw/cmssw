#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESHandleExceptionFactory.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <exception>

namespace edm {

  eventsetup::ComponentDescription const* ESHandleBase::description() const {
    if (!description_) {
      throw edm::Exception(edm::errors::InvalidReference, "NullPointer");
    }
    return description_;
  }
  void ESHandleBase::throwIfDataNotAvailable() {
    throw edm::Exception(edm::errors::InvalidReference, "ESHandle was not set.");
  }
}  // namespace edm
