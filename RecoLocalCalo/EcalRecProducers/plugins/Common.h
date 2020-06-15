#ifndef RecoLocalCalo_EcalRecProducers_plugins_Common_h
#define RecoLocalCalo_EcalRecProducers_plugins_Common_h

#include <cstdint>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

// a workaround for std::abs not being a constexpr function
namespace ecal {

  template <typename T>
  constexpr T abs(T const& value) {
    return ::std::max(value, -value);
  }

  // temporary
  namespace mgpa {

    constexpr int adc(uint16_t sample) { return sample & 0xfff; }
    constexpr int gainId(uint16_t sample) { return (sample >> 12) & 0x3; }

  }  // namespace mgpa

}  // namespace ecal

template <typename T>
struct DurationMeasurer {
  DurationMeasurer(std::string const& msg) : msg_{msg}, start_{std::chrono::high_resolution_clock::now()} {}

  ~DurationMeasurer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<T>(end - start_).count();
    std::cout << msg_ << "\nduration = " << duration << std::endl;
  }

private:
  std::string msg_;
  std::chrono::high_resolution_clock::time_point start_;
};

#endif  // RecoLocalCalo_EcalRecProducers_plugins_Common_h
