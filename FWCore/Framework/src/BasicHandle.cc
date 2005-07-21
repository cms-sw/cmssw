#include "FWCore/Framework/interface/BasicHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  void
  BasicHandle::nullHandle() const {
    throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
      << "edm::BasicHandle has null pointer to Wrapper" << '\n';
  }

  void
  BasicHandle::invalidHandle(char const* typeName) const {
    throw edm::Exception(edm::errors::LogicError,"ConvertType")
      << "edm::Wrapper converting from EDProduct to "
      << typeName << '\n';
  }
}
