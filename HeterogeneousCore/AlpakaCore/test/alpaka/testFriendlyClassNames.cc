#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"

namespace {

  template <typename T>
  std::string getFriendlyName() {
    return edm::friendlyname::friendlyName(edm::typeDemangle(typeid(T).name()));
  }

}  // namespace

TEST_CASE("Test edm::friendlyname::friendlyName for alpaka types ", "edm::friendlyname::friendlyName") {
  SECTION("CPU") {
    REQUIRE(getFriendlyName<alpaka::DevCpu>() == "alpakaDevCpu");
    REQUIRE(getFriendlyName<alpaka::QueueCpuBlocking>() == "alpakaQueueCpuBlocking");
    REQUIRE(getFriendlyName<alpaka::QueueCpuNonBlocking>() == "alpakaQueueCpuNonBlocking");
  }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  SECTION("CUDA") {
    REQUIRE(getFriendlyName<alpaka::DevCudaRt>() == "alpakaDevCudaRt");
    REQUIRE(getFriendlyName<alpaka::QueueCudaRtBlocking>() == "alpakaQueueCudaRtBlocking");
    REQUIRE(getFriendlyName<alpaka::QueueCudaRtNonBlocking>() == "alpakaQueueCudaRtNonBlocking");
  }
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  SECTION("ROCm") {
    REQUIRE(getFriendlyName<alpaka::DevHipRt>() == "alpakaDevHipRt");
    REQUIRE(getFriendlyName<alpaka::QueueHipRtBlocking>() == "alpakaQueueHipRtBlocking");
    REQUIRE(getFriendlyName<alpaka::QueueHipRtNonBlocking>() == "alpakaQueueHipRtNonBlocking");
  }
#endif
}
