#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  void
  RefCore::badID() const {
    throw edm::Exception(errors::InvalidReference,"BadID")
      << "RefCore::RefCore: Ref initialized with zero id.";
  }
}
