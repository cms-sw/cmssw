#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <numbers>
#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct phiFuncsUnitTestsKernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T* out) const {
    // Unit circle typical values
    out[0] = phi<TAcc, T>(acc, 1.0, 0.0);          // x = 1.0, y = 0.0 => phi = 0
    out[1] = phi<TAcc, T>(acc, 0.0, 1.0);          // x = 0.0, y = 1.0 => phi = pi/2
    out[2] = phi<TAcc, T>(acc, -1.0, 0.0);         // x = -1.0, y = 0.0 => phi = pi
    out[3] = phi<TAcc, T>(acc, 0.0, -1.0);         // x = 0.0, y = -1.0 => phi = -pi/2
    out[4] = phi<TAcc, T>(acc, 0.7071, 0.7071);    // x = sqrt(2)/2, y = sqrt(2)/2 => phi = pi/4
    out[5] = phi<TAcc, T>(acc, -0.7071, -0.7071);  // x = sqrt(2)/2, y = sqrt(2)/2 => phi = -3pi/4

    // Making sure that delta phi is within [-pi, pi] range
    // Phi from unit circle
    out[6] = deltaPhi<TAcc, T>(acc, 1.0, 0.0, 0.0, -1.0);                // 3pi/2 - 0 = -pi/2
    out[7] = deltaPhi<TAcc, T>(acc, 0.0, 1.0, 0.0, -1.0);                // 3pi/2 - pi/2 = pi
    out[8] = deltaPhi<TAcc, T>(acc, 0.0, -1.0, 0.0, 1.0);                // pi/2 - 3pi/2 = -pi
    out[9] = deltaPhi<TAcc, T>(acc, 0.7071, -0.7071, -0.7071, -0.7071);  // -3pi/4 - (-pi/4) = -pi/2

    // Calculation directly from phi
    out[10] = deltaPhi<TAcc, T>(acc, 3. * M_PI / 2., 0.);           // 3pi/2 - 0 = -pi/2
    out[11] = deltaPhi<TAcc, T>(acc, 3. * M_PI / 2., M_PI / 2.);    // 3pi/2 - pi/2 = pi
    out[12] = deltaPhi<TAcc, T>(acc, M_PI / 2., 3. * M_PI / 2.);    // pi/2 - 3pi/2 = -pi
    out[13] = deltaPhi<TAcc, T>(acc, -3. * M_PI / 4., -M_PI / 4.);  // -3pi/4 - (-pi/4) = -pi/2
  }
};

template <typename T>
void testPhiFuncs(uint32_t size, std::vector<double> const& res) {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.");
  }

  for (auto const& device : devices) {
    std::cout << "...on " << alpaka::getName(device) << "\n";
    Queue queue(device);

    auto c_h = make_host_buffer<T[]>(queue, size);
    alpaka::memset(queue, c_h, 0.);
    auto c_d = make_device_buffer<T[]>(queue, size);
    alpaka::memset(queue, c_d, 0.);

    alpaka::exec<Acc1D>(queue, WorkDiv1D{1u, 1u, 1u}, phiFuncsUnitTestsKernel(), c_d.data());
    alpaka::memcpy(queue, c_h, c_d);
    alpaka::wait(queue);

    constexpr T eps = 1.e-5;
    for (size_t i = 0; i < size; ++i) {
      CHECK_THAT(c_h.data()[i], Catch::Matchers::WithinAbs(res[i], eps));
    }
  }
}

TEST_CASE("Standard checks alpaka phi functions for the relevant data types (float and double) and for all backends") {
  std::vector<double> res = {0.0,
                             M_PI / 2.,
                             M_PI,
                             -M_PI / 2.,
                             M_PI / 4.,
                             -3. * M_PI / 4.,
                             -M_PI / 2.,
                             M_PI,
                             -M_PI,
                             -M_PI / 2.,
                             -M_PI / 2.,
                             M_PI,
                             -M_PI,
                             -M_PI / 2.};  // Expected results
  uint32_t size = res.size();              // Number of tests

  SECTION("Tests for double data type") {
    std::cout << "Testing phi functions for double data type...\n";
    testPhiFuncs<double>(size, res);
  }

  SECTION("Tests for float data type") {
    std::cout << "Testing phi functions for float data type...\n";
    testPhiFuncs<float>(size, res);
  }
}
