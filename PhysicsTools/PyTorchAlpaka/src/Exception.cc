#include "PhysicsTools/PyTorchAlpaka/interface/Exception.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace cms::torch::alpakatools::detail {

  [[noreturn]] void throwException(const std::string& category, const std::string& message) {
    throw cms::Exception(category) << message;
  }

}  // namespace cms::torch::alpakatools::detail
