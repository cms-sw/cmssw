#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <cuda_runtime.h>

#include "HeterogeneousTest/CUDADevice/interface/DeviceAddition.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

__global__ void kernel_add_vectors_f(const float* __restrict__ in1,
                                     const float* __restrict__ in2,
                                     float* __restrict__ out,
                                     size_t size) {
  cms::cudatest::add_vectors_f(in1, in2, out, size);
}

TEST_CASE("HeterogeneousTest/CUDADevice test", "[cudaTestDeviceAddition]") {
  cms::cudatest::requireDevices();

  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine rand{rd()};
  std::normal_distribution<float> dist{0., 1.};

  // tolerance
  constexpr float epsilon = 0.000001;

  // buffer size
  constexpr size_t size = 1024 * 1024;

  // allocate input and output host buffers
  std::vector<float> in1_h(size);
  std::vector<float> in2_h(size);
  std::vector<float> out_h(size);

  // fill the input buffers with random data, and the output buffer with zeros
  for (size_t i = 0; i < size; ++i) {
    in1_h[i] = dist(rand);
    in2_h[i] = dist(rand);
    out_h[i] = 0.;
  }

  SECTION("Test add_vectors_f") {
    // allocate input and output buffers on the device
    float* in1_d;
    float* in2_d;
    float* out_d;
    REQUIRE_NOTHROW(cudaCheck(cudaMalloc(&in1_d, size * sizeof(float))));
    REQUIRE_NOTHROW(cudaCheck(cudaMalloc(&in2_d, size * sizeof(float))));
    REQUIRE_NOTHROW(cudaCheck(cudaMalloc(&out_d, size * sizeof(float))));

    // copy the input data to the device
    REQUIRE_NOTHROW(cudaCheck(cudaMemcpy(in1_d, in1_h.data(), size * sizeof(float), cudaMemcpyHostToDevice)));
    REQUIRE_NOTHROW(cudaCheck(cudaMemcpy(in2_d, in2_h.data(), size * sizeof(float), cudaMemcpyHostToDevice)));

    // fill the output buffer with zeros
    REQUIRE_NOTHROW(cudaCheck(cudaMemset(out_d, 0, size * sizeof(float))));

    // launch the 1-dimensional kernel for vector addition
    kernel_add_vectors_f<<<32, 32>>>(in1_d, in2_d, out_d, size);
    REQUIRE_NOTHROW(cudaCheck(cudaGetLastError()));

    // copy the results from the device to the host
    REQUIRE_NOTHROW(cudaCheck(cudaMemcpy(out_h.data(), out_d, size * sizeof(float), cudaMemcpyDeviceToHost)));

    // wait for all the operations to complete
    REQUIRE_NOTHROW(cudaCheck(cudaDeviceSynchronize()));

    // check the results
    for (size_t i = 0; i < size; ++i) {
      float sum = in1_h[i] + in2_h[i];
      CHECK_THAT(out_h[i], Catch::Matchers::WithinAbs(sum, epsilon));
    }
  }
}
