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
    for (auto index : cms::alpakatools::uniform_elements(acc, size)) {
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernelSkip {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ in1,
                                T const* __restrict__ in2,
                                T* __restrict__ out,
                                size_t first,
                                size_t size) const {
    for (auto index : cms::alpakatools::uniform_elements(acc, first, size)) {
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernel1D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, Vec1D size) const {
    for (auto ndindex : cms::alpakatools::uniform_elements_nd(acc, size)) {
      auto index = ndindex[0];
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernel2D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, Vec2D size) const {
    for (auto ndindex : cms::alpakatools::uniform_elements_nd(acc, size)) {
      auto index = ndindex[0] * size[1] + ndindex[1];
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernel3D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, Vec3D size) const {
    for (auto ndindex : cms::alpakatools::uniform_elements_nd(acc, size)) {
      auto index = (ndindex[0] * size[1] + ndindex[1]) * size[2] + ndindex[2];
      out[index] = in1[index] + in2[index];
    }
  }
};

/* This is not an efficient approach; it is only a test of using dynamic shared memory,
 * split block and element loops, and block-level synchronisation
 */

struct VectorAddBlockKernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, size_t size) const {
    // block size
    auto const blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
    // get the dynamic shared memory buffer
    T* buffer = alpaka::getDynSharedMem<T>(acc);
    // the outer loop is needed to repeat the "block" as many times as needed to cover the whole problem space
    // the inner loop is needed for backends that use more than one element per thread
    for (auto block : cms::alpakatools::uniform_groups(acc, size)) {
      // only one thread per block: initialise the shared memory
      if (cms::alpakatools::once_per_block(acc)) {
        // not really necessary, just to show how to use "once_per_block"
        for (Idx local = 0; local < blockSize; ++local)
          buffer[local] = 0.;
      }
      // synchronise all threads in the block
      alpaka::syncBlockThreads(acc);
      // read the first set of data into shared memory
      for (auto index : cms::alpakatools::uniform_group_elements(acc, block, size)) {
        buffer[index.local] = in1[index.global];
      }
      // synchronise all threads in the block
      alpaka::syncBlockThreads(acc);
      // add the second set of data into shared memory
      for (auto index : cms::alpakatools::uniform_group_elements(acc, block, size)) {
        buffer[index.local] += in2[index.global];
      }
      // synchronise all threads in the block
      alpaka::syncBlockThreads(acc);
      // store the results into global memory
      for (auto index : cms::alpakatools::uniform_group_elements(acc, block, size)) {
        out[index.global] = buffer[index.local];
      }
    }
  }
};

/* Run all operations in a single thread.
 * Written in an inefficient way to test "once_per_grid".
 */

struct VectorAddKernelSerial {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, size_t size) const {
    // the operations are performed by a single thread
    if (cms::alpakatools::once_per_grid(acc)) {
      for (Idx index = 0; index < size; ++index) {
        out[index] += in1[index];
        out[index] += in2[index];
      }
    }
  }
};

/* Run all operations in one thread per block.
 * Written in an inefficient way to test "once_per_block".
 */

struct VectorAddKernelBlockSerial {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, T const* __restrict__ in1, T const* __restrict__ in2, T* __restrict__ out, size_t size) const {
    // block size
    auto const blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
    // the loop is used to repeat the "block" as many times as needed to cover the whole problem space
    for (auto block : cms::alpakatools::uniform_groups(acc, size)) {
      // the operations are performed by a single thread in each "logical" block
      const auto first = blockSize * block;
      const auto range = std::min<size_t>(first + blockSize, size);
      if (cms::alpakatools::once_per_block(acc)) {
        for (Idx index = first; index < range; ++index) {
          out[index] += in1[index];
          out[index] += in2[index];
        }
      }
    }
  }
};

namespace alpaka::trait {
  // specialize the BlockSharedMemDynSizeBytes trait to specify the amount of
  // block shared dynamic memory for the VectorAddBlockKernel kernel
  template <typename TAcc>
  struct BlockSharedMemDynSizeBytes<VectorAddBlockKernel, TAcc> {
    // the size in bytes of the shared memory allocated for a block
    template <typename T>
    ALPAKA_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(VectorAddBlockKernel const& /* kernel */,
                                                                        Vec1D threads,
                                                                        Vec1D elements,
                                                                        T const* __restrict__ /* in1 */,
                                                                        T const* __restrict__ /* in2 */,
                                                                        T* __restrict__ /* out */,
                                                                        size_t size) {
      return static_cast<std::size_t>(threads[0] * elements[0] * sizeof(T));
    }
  };
}  // namespace alpaka::trait

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

// test the 1-dimensional kernel on all devices, potentially skipping some elements
template <typename TKernel>
void testVectorAddKernelSkip(std::size_t skip_elements,
                             std::size_t problem_size,
                             std::size_t grid_size,
                             std::size_t block_size,
                             TKernel kernel) {
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
    std::cout << "Test 1D vector addition on " << alpaka::getName(device) << " skipping " << skip_elements << " over "
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

    // launch the 1-dimensional kernel with scalar size
    auto div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::exec<Acc1D>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), skip_elements, size);

    // copy the results from the device to the host
    alpaka::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::wait(queue);

    // check the results
    for (size_t i = 0; i < skip_elements; ++i) {
      REQUIRE(out_h[i] == 0);
    }
    for (size_t i = skip_elements; i < size; ++i) {
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

    // launch the 1-dimensional kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test 1D vector block-level addition with small block size, using scalar dimensions\n";
    testVectorAddKernel(10000, 32, 32, VectorAddBlockKernel{});

    // launch the 1-dimensional kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test 1D vector block-level addition with large block size, using scalar dimensions\n";
    testVectorAddKernel(100, 1, 1024, VectorAddBlockKernel{});

    // launch the 1-dimensional kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test 1D vector single-threaded serial addition with small block size, using scalar dimensions\n";
    testVectorAddKernel(10000, 32, 32, VectorAddKernelSerial{});

    // launch the 1-dimensional kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test 1D vector single-threaded seria addition with large block size, using scalar dimensions\n";
    testVectorAddKernel(100, 1, 1024, VectorAddKernelSerial{});

    // launch the 1-dimensional kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test 1D vector block-level serial addition with small block size, using scalar dimensions\n";
    testVectorAddKernel(10000, 32, 32, VectorAddKernelBlockSerial{});

    // launch the 1-dimensional kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test 1D vector block-level serial addition with large block size, using scalar dimensions\n";
    testVectorAddKernel(100, 1, 1024, VectorAddKernelBlockSerial{});

    // launch the 1-dimensional kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test 1D vector addition with small block size, using scalar dimensions\n";
    testVectorAddKernelSkip(20, 10000, 32, 32, VectorAddKernelSkip{});

    // launch the 1-dimensional kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test 1D vector addition with large block size, using scalar dimensions\n";
    testVectorAddKernelSkip(20, 100, 1, 1024, VectorAddKernelSkip{});
  }
}
