#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/Exception.h"

namespace cms::Ort::alpakatools::detail {

  [[noreturn]] void throwException(const std::string& category, const std::string& message) {
    throw cms::Exception(category) << message;
  }

  [[noreturn]] void throwException(const std::string& category,
                                   const std::string& message,
                                   const std::string& context) {
    cms::Exception ex(category);
    ex << message;
    ex.addContext(context);
    throw ex;
  }

}  // namespace cms::Ort::alpakatools::detail
