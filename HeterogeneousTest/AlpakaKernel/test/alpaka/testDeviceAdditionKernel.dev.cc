#include <cstdint>
#include <random>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousTest/AlpakaKernel/interface/alpaka/DeviceAdditionKernel.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

TEST_CASE("HeterogeneousTest/AlpakaKernel test", "[alpakaTestDeviceAdditionKernel]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
        "the test will be skipped.");
  }

  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine rand{rd()};
  std::normal_distribution<float> dist{0., 1.};

  // tolerance
  constexpr float epsilon = 0.000001;

  // buffer size
  constexpr uint32_t size = 1024 * 1024;

  // allocate input and output host buffers
  std::vector<float> in1_h(size);
  std::vector<float> in2_h(size);
  std::vector<float> out_h(size);

  // fill the input buffers with random data, and the output buffer with zeros
  for (uint32_t i = 0; i < size; ++i) {
    in1_h[i] = dist(rand);
    in2_h[i] = dist(rand);
    out_h[i] = 0.;
  }

  // run the test on all available devices
  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    SECTION("Test add_vectors_f on " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend") {
      REQUIRE_NOTHROW([&]() {
        Queue queue{device};

        // allocate input and output buffers on the device
        auto in1_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
        auto in2_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
        auto out_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);

        // copy the input data to the device
        // FIXME: pass the explicit size of type uint32_t to avoid compilation error
        // The destination view and the extent are required to have compatible index types!
        alpaka::memcpy(queue, in1_d, in1_h, size);
        alpaka::memcpy(queue, in2_d, in2_h, size);

        // fill the output buffer with zeros
        alpaka::memset(queue, out_d, 0);

        // launch the 1-dimensional kernel for vector addition
        alpaka::exec<Acc1D>(queue,
                            cms::alpakatools::make_workdiv<Acc1D>(32, 32),
                            test::KernelAddVectorsF{},
                            in1_d.data(),
                            in2_d.data(),
                            out_d.data(),
                            size);

        // copy the results from the device to the host
        alpaka::memcpy(queue, out_h, out_d, size);

        // wait for all the operations to complete
        alpaka::wait(queue);
      }());

      // check the results
      for (uint32_t i = 0; i < size; ++i) {
        float sum = in1_h[i] + in2_h[i];
        CHECK_THAT(out_h[i], Catch::Matchers::WithinAbs(sum, epsilon));
      }
    }
  }
}
