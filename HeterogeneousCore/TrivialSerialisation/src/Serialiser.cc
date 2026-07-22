#include "FWCore/Utilities/interface/EDMException.h"

namespace ngt::detail {

  void throwEmptyWrapperError() {
    throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
  }

  void throwHostProductTypeIDError() {
    throw edm::Exception(edm::errors::LogicError)
        << "hostProductTypeID() called on a type without a CopyToHost<T> specialisation";
  }

}  // namespace ngt::detail
