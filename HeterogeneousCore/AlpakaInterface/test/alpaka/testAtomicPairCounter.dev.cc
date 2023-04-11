#include <cassert>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

static constexpr auto s_tag = "[" ALPAKA_TYPE_ALIAS_NAME(alpakaTestAtomicPair) "]";

struct update {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      const TAcc &acc, AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t i) {
      auto m = i % 11;
      m = m % 6 + 1;  // max 6, no 0
      auto c = dc->add(acc, m);
      assert(c.m < n);
      ind[c.m] = c.n;
      for (uint32_t j = c.n; j < c.n + m; ++j)
        cont[j] = i;
    });
  }
};

struct finalize {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      const TAcc &acc, AtomicPairCounter const *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    assert(dc->get().m == n);
    ind[n] = dc->get().n;
  }
};

TEST_CASE("Standard checks of " ALPAKA_TYPE_ALIAS_NAME(alpakaTestAtomicPair), s_tag) {
    SECTION("AtomicPairCounter") {

        auto const& devices = cms::alpakatools::devices<Platform>();
        if (devices.empty()) {
        std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
                    << ", the test will be skipped.\n";
        return;
        }
    // run the test on each device
    for (auto const& device : devices) {
        std::cout << "Test AtomicPairCounter on " << alpaka::getName(device) << '\n';
        auto queue = Queue(device);

        auto c_d = make_device_buffer<AtomicPairCounter>(queue);
        alpaka::memset(queue, c_d, 0);

        std::cout << "- size " << sizeof(AtomicPairCounter) << std::endl;

        constexpr uint32_t N = 20000;
        constexpr uint32_t M = N * 6;
        auto n_d = make_device_buffer<uint32_t[]>(queue, N);
        auto m_d = make_device_buffer<uint32_t[]>(queue, M);

        constexpr uint32_t NUM_VALUES = 10000;

        // Update
        const auto blocksPerGrid = 2000u;
        const auto threadsPerBlockOrElementsPerThread = 512u;
        const auto workDiv = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDiv, update(), c_d.data(), n_d.data(), m_d.data(), NUM_VALUES));

        // Finalize
        const auto blocksPerGridFinalize = 1u;
        const auto threadsPerBlockOrElementsPerThreadFinalize = 1u;
        const auto workDivFinalize = make_workdiv<Acc1D>(blocksPerGridFinalize, threadsPerBlockOrElementsPerThreadFinalize);
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc1D>(workDivFinalize, finalize(), c_d.data(), n_d.data(), m_d.data(), NUM_VALUES));
        
        auto c_h = make_host_buffer<AtomicPairCounter>(queue);
        auto n_h = make_host_buffer<uint32_t[]>(queue, N);
        auto m_h = make_host_buffer<uint32_t[]>(queue, M);

        // copy the results from the device to the host
        alpaka::memcpy(queue, c_h, c_d);
        alpaka::memcpy(queue, n_h, n_d);
        alpaka::memcpy(queue, m_h, m_d);

        // wait for all the operations to complete
        alpaka::wait(queue);

        REQUIRE(c_h.data()->get().m == NUM_VALUES);
        REQUIRE(n_h[NUM_VALUES] == c_h.data()->get().n);
        REQUIRE(n_h[0] == 0);
        
        for (size_t i = 0; i < NUM_VALUES; ++i) {
            auto ib = n_h.data()[i];
            auto ie = n_h.data()[i + 1];
            auto k = m_h.data()[ib++];
            REQUIRE(k < NUM_VALUES);

            for (; ib < ie; ++ib)
              REQUIRE(m_h.data()[ib] == k);
          
        }

    }
    }
}