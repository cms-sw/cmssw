#ifndef HeterogeneousCore_SonicCore_sonic_utils
#define HeterogeneousCore_SonicCore_sonic_utils

#include <string_view>
#include <chrono>

namespace sonic_utils {
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

  void printDebugTime(std::string_view debugName, std::string_view msg, const TimePoint& t0);
}  // namespace sonic_utils

#endif
