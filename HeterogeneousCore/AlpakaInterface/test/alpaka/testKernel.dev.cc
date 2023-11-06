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

// test the 1-dimensional kernel on all devices
template <typename TKernel>
void testVectorAddKernel(std::size_t problem_size, std::size_t grid_size, std::size_t block_size, TKernel kernel) {
  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine rand{rd()};
  std::normal_distribution<float> dist{0., 1.};

  // tolerance
  constexpr float epsilon = 0.000001;

  // buffer size
  const size_t size = problem_size;

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
  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    std::cout << "Test 1D vector addition on " << alpaka::getName(device) << " over " << problem_size << " values with "
              << grid_size << " blocks of " << block_size << " elements\n";
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
    auto div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::exec<Acc1D>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), size);

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

// test the N-dimensional kernels on all devices
template <typename TDim, typename TKernel>
void testVectorAddKernelND(Vec<TDim> problem_size, Vec<TDim> grid_size, Vec<TDim> block_size, TKernel kernel) {
  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine rand{rd()};
  std::normal_distribution<float> dist{0., 1.};

  // tolerance
  constexpr float epsilon = 0.000001;

  // linearised buffer size
  const size_t size = problem_size.prod();

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
  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    std::cout << "Test " << TDim::value << "D vector addition on " << alpaka::getName(device) << " over "
              << problem_size << " values with " << grid_size << " blocks of " << block_size << " elements\n";
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
    using AccND = Acc<TDim>;
    auto div = cms::alpakatools::make_workdiv<AccND>(grid_size, block_size);
    alpaka::exec<AccND>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), problem_size);

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

TEST_CASE("Test alpaka kernels for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  SECTION("Alpaka N-dimensional kernels") {
    // get the list of devices on the current platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty()) {
      INFO("No devices available on the platform " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE));
      REQUIRE(not devices.empty());
    }

    // launch the 1-dimensional kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test 1D vector addition with small block size, using scalar dimensions\n";
    testVectorAddKernel(10000, 32, 32, VectorAddKernel{});

    // launch the 1-dimensional kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test 1D vector addition with large block size, using scalar dimensions\n";
    testVectorAddKernel(100, 1, 1024, VectorAddKernel{});

    // launch the 1-dimensional kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test 1D vector addition with small block size\n";
    testVectorAddKernelND<Dim1D>({10000}, {32}, {32}, VectorAddKernel1D{});

    // launch the 1-dimensional kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test 1D vector addition with large block size\n";
    testVectorAddKernelND<Dim1D>({100}, {1}, {1024}, VectorAddKernel1D{});

    // launch the 2-dimensional kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test 2D vector addition with small block size\n";
    testVectorAddKernelND<Dim2D>({400, 250}, {4, 4}, {16, 16}, VectorAddKernel2D{});

    // launch the 2-dimensional kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test 2D vector addition with large block size\n";
    testVectorAddKernelND<Dim2D>({20, 20}, {1, 1}, {32, 32}, VectorAddKernel2D{});

    // launch the 3-dimensional kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test 3D vector addition with small block size\n";
    testVectorAddKernelND<Dim3D>({50, 125, 16}, {5, 5, 1}, {4, 4, 4}, VectorAddKernel3D{});

    // launch the 3-dimensional kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test 3D vector addition with large block size\n";
    testVectorAddKernelND<Dim3D>({5, 5, 5}, {1, 1, 1}, {8, 8, 8}, VectorAddKernel3D{});
  }
}
