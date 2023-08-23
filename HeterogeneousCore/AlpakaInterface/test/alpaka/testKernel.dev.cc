#include <cstdio>
#include <random>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// each test binary is built for a single Alpaka backend
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

static constexpr auto s_tag = "[" ALPAKA_TYPE_ALIAS_NAME(alpakaTestKernel) "]";

struct VectorAddKernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, size_t size) const {
    for (auto index : cms::alpakatools::elements_with_stride(acc, size)) {
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernel1D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, Vec1D size) const {
    for (auto ndindex : cms::alpakatools::elements_with_stride_nd(acc, size)) {
      auto index = ndindex[0];
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernel2D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, Vec2D size) const {
    for (auto ndindex : cms::alpakatools::elements_with_stride_nd(acc, size)) {
      auto index = ndindex[0] * size[1] + ndindex[1];
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernel3D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, Vec3D size) const {
    for (auto ndindex : cms::alpakatools::elements_with_stride_nd(acc, size)) {
      auto index = (ndindex[0] * size[1] + ndindex[1]) * size[2] + ndindex[2];
      out[index] = in1[index] + in2[index];
    }
  }
};

TEST_CASE("Standard checks of " ALPAKA_TYPE_ALIAS_NAME(alpakaTestKernel), s_tag) {
  SECTION("VectorAddKernel") {
    // get the list of devices on the current platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty()) {
      std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
                << ", the test will be skipped.\n";
      return;
    }

    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0., 1.};

    // tolerance
    constexpr float epsilon = 0.000001;

    // buffer size
    constexpr size_t size = 1024 * 1024;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);
    auto in2_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);
    auto out_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);

    // fill the input buffers with random data, and the output buffer with zeros
    for (size_t i = 0; i < size; ++i) {
      in1_h[i] = dist(rand);
      in2_h[i] = dist(rand);
      out_h[i] = 0.;
    }

    // run the test on each device
    for (auto const& device : devices) {
      std::cout << "Test 1D vector addition on " << alpaka::getName(device) << '\n';
      auto queue = Queue(device);

      // allocate input and output buffers on the device
      auto in1_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
      auto in2_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
      auto out_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);

      // copy the input data to the device; the size is known from the buffer objects
      alpaka::memcpy(queue, in1_d, in1_h);
      alpaka::memcpy(queue, in2_d, in2_h);

      // fill the output buffer with zeros; the size is known from the buffer objects
      alpaka::memset(queue, out_d, 0.);

      // launch the 1-dimensional kernel with scalar size
      auto div = cms::alpakatools::make_workdiv<Acc1D>(4, 4);
      alpaka::exec<Acc1D>(queue, div, VectorAddKernel{}, in1_d.data(), in2_d.data(), out_d.data(), size);

      // copy the results from the device to the host
      alpaka::memcpy(queue, out_h, out_d);

      // wait for all the operations to complete
      alpaka::wait(queue);

      // check the results
      for (size_t i = 0; i < size; ++i) {
        float sum = in1_h[i] + in2_h[i];
        REQUIRE(out_h[i] < sum + epsilon);
        REQUIRE(out_h[i] > sum - epsilon);
      }

      // reset the output buffer on the device to all zeros
      alpaka::memset(queue, out_d, 0.);

      // launch the 1-dimensional kernel with vector size
      alpaka::exec<Acc1D>(queue, div, VectorAddKernel1D{}, in1_d.data(), in2_d.data(), out_d.data(), size);

      // copy the results from the device to the host
      alpaka::memcpy(queue, out_h, out_d);

      // wait for all the operations to complete
      alpaka::wait(queue);

      // check the results
      for (size_t i = 0; i < size; ++i) {
        float sum = in1_h[i] + in2_h[i];
        REQUIRE(out_h[i] < sum + epsilon);
        REQUIRE(out_h[i] > sum - epsilon);
      }
    }
  }
}

TEST_CASE("Standard checks of " ALPAKA_TYPE_ALIAS_NAME(alpakaTestKernel2D), s_tag) {
  SECTION("VectorAddKernel2D") {
    // get the list of devices on the current platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty()) {
      std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
                << ", the test will be skipped.\n";
      return;
    }

    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0., 1.};

    // tolerance
    constexpr float epsilon = 0.000001;

    // 3-dimensional and linearised buffer size
    constexpr Vec2D ndsize = {16, 16};
    constexpr size_t size = ndsize.prod();

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);
    auto in2_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);
    auto out_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);

    // fill the input buffers with random data, and the output buffer with zeros
    for (size_t i = 0; i < size; ++i) {
      in1_h[i] = dist(rand);
      in2_h[i] = dist(rand);
      out_h[i] = 0.;
    }

    // run the test on each device
    for (auto const& device : devices) {
      std::cout << "Test 2D vector addition on " << alpaka::getName(device) << '\n';
      auto queue = Queue(device);

      // allocate input and output buffers on the device
      auto in1_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
      auto in2_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
      auto out_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);

      // copy the input data to the device; the size is known from the buffer objects
      alpaka::memcpy(queue, in1_d, in1_h);
      alpaka::memcpy(queue, in2_d, in2_h);

      // fill the output buffer with zeros; the size is known from the buffer objects
      alpaka::memset(queue, out_d, 0.);

      // launch the 3-dimensional kernel
      auto div = cms::alpakatools::make_workdiv<Acc2D>({4, 4}, {32, 32});
      alpaka::exec<Acc2D>(queue, div, VectorAddKernel2D{}, in1_d.data(), in2_d.data(), out_d.data(), ndsize);

      // copy the results from the device to the host
      alpaka::memcpy(queue, out_h, out_d);

      // wait for all the operations to complete
      alpaka::wait(queue);

      // check the results
      for (size_t i = 0; i < size; ++i) {
        float sum = in1_h[i] + in2_h[i];
        REQUIRE(out_h[i] < sum + epsilon);
        REQUIRE(out_h[i] > sum - epsilon);
      }
    }
  }
}

TEST_CASE("Standard checks of " ALPAKA_TYPE_ALIAS_NAME(alpakaTestKernel3D), s_tag) {
  SECTION("VectorAddKernel3D") {
    // get the list of devices on the current platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty()) {
      std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
                << ", the test will be skipped.\n";
      return;
    }

    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0., 1.};

    // tolerance
    constexpr float epsilon = 0.000001;

    // 3-dimensional and linearised buffer size
    constexpr Vec3D ndsize = {50, 125, 16};
    constexpr size_t size = ndsize.prod();

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);
    auto in2_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);
    auto out_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);

    // fill the input buffers with random data, and the output buffer with zeros
    for (size_t i = 0; i < size; ++i) {
      in1_h[i] = dist(rand);
      in2_h[i] = dist(rand);
      out_h[i] = 0.;
    }

    // run the test on each device
    for (auto const& device : devices) {
      std::cout << "Test 3D vector addition on " << alpaka::getName(device) << '\n';
      auto queue = Queue(device);

      // allocate input and output buffers on the device
      auto in1_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
      auto in2_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
      auto out_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);

      // copy the input data to the device; the size is known from the buffer objects
      alpaka::memcpy(queue, in1_d, in1_h);
      alpaka::memcpy(queue, in2_d, in2_h);

      // fill the output buffer with zeros; the size is known from the buffer objects
      alpaka::memset(queue, out_d, 0.);

      // launch the 3-dimensional kernel
      auto div = cms::alpakatools::make_workdiv<Acc3D>({5, 5, 1}, {4, 4, 4});
      alpaka::exec<Acc3D>(queue, div, VectorAddKernel3D{}, in1_d.data(), in2_d.data(), out_d.data(), ndsize);

      // copy the results from the device to the host
      alpaka::memcpy(queue, out_h, out_d);

      // wait for all the operations to complete
      alpaka::wait(queue);

      // check the results
      for (size_t i = 0; i < size; ++i) {
        float sum = in1_h[i] + in2_h[i];
        REQUIRE(out_h[i] < sum + epsilon);
        REQUIRE(out_h[i] > sum - epsilon);
      }
    }
  }
}
