#include "DataFormats/Common/interface/ConvertHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"

namespace edm {
  namespace handleimpl {
    static std::shared_ptr<HandleExceptionFactory> s_invalidRefFactory =
        makeHandleExceptionFactory([]() -> std::shared_ptr<cms::Exception> {
          std::shared_ptr<cms::Exception> whyFailed =
              std::make_shared<edm::Exception>(errors::InvalidReference, "NullPointer");
          *whyFailed << "Handle has null pointer to data product";
          return whyFailed;
        });

    std::shared_ptr<HandleExceptionFactory> makeInvalidReferenceException() { return s_invalidRefFactory; }

    void throwConvertTypeError(std::type_info const& expected, std::type_info const& actual) {
      throw Exception(errors::LogicError, "TypeMismatch")
          << "edm::BasicHandle contains a product of type " << actual.name() << ".\n"
          << "A type of " << expected.name() << "was expected.";
    }
  }  // namespace handleimpl
}  // namespace edm
