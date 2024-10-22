#include "HeterogeneousCore/SonicCore/interface/sonic_utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string_view>
#include <chrono>

namespace sonic_utils {
  void printDebugTime(std::string_view debugName, std::string_view msg, const TimePoint& t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    if (debugName.empty())
      return;
    edm::LogInfo(debugName) << msg << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  }
}  // namespace sonic_utils
