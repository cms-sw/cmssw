#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  class EDProduct;
  namespace handleimpl {
    void throwInvalidReference() {
      throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
        << "edm::BasicHandle has null pointer to Wrapper";
    }

    void throwConvertTypeError(EDProduct const* product) {
      throw edm::Exception(edm::errors::LogicError,"ConvertType")
        << "edm::Wrapper converting from EDProduct to "
        << typeid(*product).name();
    }
  }
}
