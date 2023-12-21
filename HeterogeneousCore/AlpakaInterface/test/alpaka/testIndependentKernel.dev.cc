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

/* Add the group id to te value of each element in the group.
 * Each group is composed by the elements first[group]..first[group+1]-1 .
 */
struct IndependentWorkKernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ in,
                                T* __restrict__ out,
                                size_t const* __restrict__ indices,
                                size_t groups) const {
    for (auto group : cms::alpakatools::independent_groups(acc, groups)) {
      size_t first = indices[group];
      size_t last = indices[group + 1];
      size_t size = last - first;
      for (auto index : cms::alpakatools::independent_group_elements(acc, size)) {
        out[first + index] = in[first + index] + group;
      }
    }
  }
};

/* Test the IndependentWorkKernel kernel on all devices
 */
template <typename TKernel>
void testIndependentWorkKernel(size_t groups, size_t grid_size, size_t block_size, TKernel kernel) {
  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine engine{rd()};

  // uniform distribution
  std::uniform_int_distribution<size_t> random_size{100, 201};

  // gaussian distribution
  std::normal_distribution<float> dist{0., 1.};

  // build the groups
  std::vector<size_t> sizes(groups);
  auto indices_h = cms::alpakatools::make_host_buffer<size_t[], Platform>(groups + 1);
  indices_h[0] = 0;
  for (size_t i = 0; i < groups; ++i) {
    auto size = random_size(engine);
    sizes[i] = size;
    indices_h[i + 1] = indices_h[i] + size;
  }

  // tolerance
  constexpr float epsilon = 0.000001;

  // buffer size
  const size_t size = indices_h[groups];

  // allocate the input and output host buffer in pinned memory accessible by the Platform devices
  auto in_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);
  auto out_h = cms::alpakatools::make_host_buffer<float[], Platform>(size);

  // fill the input buffers with random data, and the output buffer with zeros
  for (size_t i = 0; i < size; ++i) {
    in_h[i] = dist(engine);
    out_h[i] = 0;
  }

  // run the test on each device
  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    std::cout << "Test IndependentWorkKernel on " << alpaka::getName(device) << " over " << size << " elements in "
              << groups << " independent groups with " << grid_size << " blocks of " << block_size << " elements\n";
    auto queue = Queue(device);

    // allocate input and output buffers on the device
    auto indices_d = cms::alpakatools::make_device_buffer<size_t[]>(queue, groups + 1);
    auto in_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
    auto out_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::memcpy(queue, indices_d, indices_h);
    alpaka::memcpy(queue, in_d, in_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::memset(queue, out_d, 0.);

    // launch the 1-dimensional kernel with independent work groups
    auto div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::exec<Acc1D>(queue, div, kernel, in_d.data(), out_d.data(), indices_d.data(), groups);

    // copy the results from the device to the host
    alpaka::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::wait(queue);

    // check the results
    for (size_t g = 0; g < groups; ++g) {
      size_t first = indices_h[g];
      size_t last = indices_h[g + 1];
      for (size_t i = first; i < last; ++i) {
        float sum = in_h[i] + g;
        float delta = std::max(std::fabs(sum) * epsilon, epsilon);
        REQUIRE(out_h[i] < sum + delta);
        REQUIRE(out_h[i] > sum - delta);
      }
    }
  }
}

TEST_CASE("Test alpaka kernels for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  SECTION("Independent work groups") {
    // get the list of devices on the current platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty()) {
      INFO("No devices available on the platform " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE));
      REQUIRE(not devices.empty());
    }

    // launch the independent work kernel with a small block size and a small number of blocks;
    // this relies on the kernel to loop over the "problem space" and do more work per block
    std::cout << "Test independent work kernel with small block size, using scalar dimensions\n";
    testIndependentWorkKernel(100, 32, 32, IndependentWorkKernel{});

    // launch the independent work kernel with a large block size and a single block;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test independent work kernel with large block size, using scalar dimensions\n";
    testIndependentWorkKernel(100, 1, 1024, IndependentWorkKernel{});

    // launch the independent work kernel with a large block size and a large number of blocks;
    // this relies on the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data
    std::cout << "Test independent work kernel with large block size, using scalar dimensions\n";
    testIndependentWorkKernel(100, 1024, 1024, IndependentWorkKernel{});
  }
}
