#include "DataFormats/Common/interface/ConvertHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace handleimpl {
    void throwInvalidReference() {
      throw Exception(errors::InvalidReference, "NullPointer")
        << "edm::BasicHandle has null pointer to Wrapper";
    }

    void throwConvertTypeError(std::type_info const& expected, std::type_info const& actual) {
      throw Exception(errors::LogicError, "TypeMismatch")
        << "edm::BasicHandle contains a product of type " << actual.name() << ".\n"
        << "A type of " << expected.name() << "was expected.";
    }
  }
}
