#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/Backend.h"

#include <algorithm>
#include <array>

namespace {
  constexpr const std::array<std::string_view, static_cast<short>(cms::alpakatools::Backend::size)> backendNames = {
      {"SerialSync", "CudaAsync", "ROCmAsync", "TbbAsync"}};
}

namespace cms::alpakatools {
  Backend toBackend(std::string_view name) {
    auto found = std::find(backendNames.begin(), backendNames.end(), name);
    if (found == backendNames.end()) {
      cms::Exception ex("EnumNotFound");
      ex << "Invalid backend name '" << name << "'";
      ex.addContext("Calling cms::alpakatools::toBackend()");
      throw ex;
    }
    return static_cast<Backend>(std::distance(backendNames.begin(), found));
  }

  std::string_view toString(Backend backend) {
    auto val = static_cast<unsigned short>(backend);
    if (val >= static_cast<unsigned short>(Backend::size)) {
      cms::Exception ex("InvalidEnumValue");
      ex << "Invalid backend enum value " << val;
      ex.addContext("Calling cms::alpakatools::toString()");
      throw ex;
    }
    return backendNames[val];
  }
}  // namespace cms::alpakatools
