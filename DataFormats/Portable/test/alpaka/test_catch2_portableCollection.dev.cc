#include <catch.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// each test binary is built for a single Alpaka backend
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  GENERATE_SOA_LAYOUT(TestLayout, SOA_COLUMN(double, x), SOA_COLUMN(int32_t, id), SOA_SCALAR(uint32_t, num))

  using TestSoA = TestLayout<>;

  constexpr auto s_tag = "[PortableCollection]";
}  // namespace

TEST_CASE("PortableCollection<T, TDev>", s_tag) {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
         "the test will be skipped.");
  }

  SECTION("Default constructor") {
    PortableHostCollection<TestSoA> coll_h;
    REQUIRE(coll_h.size() == 0);
    REQUIRE(not coll_h.isValid());

    // Following lines would be undefined behavior, and could lead to crashes
    //coll->num() = 42;
    //REQUIRE(coll->num() == 42);

    // CopyToDevice<PortableHostCollection<T>> is not defined
#ifndef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    for (auto const& device : devices) {
      auto queue = Queue(device);
      auto coll_d = cms::alpakatools::CopyToDevice<PortableHostCollection<TestSoA>>::copyAsync(queue, coll_h);
      REQUIRE(coll_d.size() == 0);
      REQUIRE(not coll_d.isValid());
      alpaka::wait(queue);
    }
#endif
  }

  SECTION("Zero size") {
    int constexpr size = 0;
    PortableHostCollection<TestSoA> coll_h(size, cms::alpakatools::host());
    REQUIRE(coll_h.isValid());
    REQUIRE(coll_h->metadata().size() == size);
    coll_h->num() = 42;

    // CopyToDevice<PortableHostCollection<T>> is not defined
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    REQUIRE(coll_h->num() == 42);
#else
    for (auto const& device : devices) {
      auto queue = Queue(device);
      auto coll_d = cms::alpakatools::CopyToDevice<PortableHostCollection<TestSoA>>::copyAsync(queue, coll_h);
      REQUIRE(coll_d.isValid());
      REQUIRE(coll_d.size() == size);

      auto div = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(
          queue,
          div,
          [] ALPAKA_FN_ACC(Acc1D const& acc, TestSoA::ConstView view) {
            assert(view.metadata().size() == size);
            assert(view.num() == 42);
          },
          coll_d.const_view());
      alpaka::wait(queue);
    }
#endif
  }

  SECTION("Non-zero size") {
    int constexpr size = 10;
    PortableHostCollection<TestSoA> coll_h(size, cms::alpakatools::host());
    REQUIRE(coll_h.isValid());
    coll_h->num() = 20;

    for (int i = 0; i < size; ++i) {
      coll_h->id(i) = i * 2 + 1;
    }

    // CopyToDevice<PortableHostCollection<T>> is not defined
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    for (int i = 0; i < size; ++i) {
      assert(coll_h->id(i) == i * 2 + 1);
    }
#else
    for (auto const& device : devices) {
      auto queue = Queue(device);
      auto coll_d = cms::alpakatools::CopyToDevice<PortableHostCollection<TestSoA>>::copyAsync(queue, coll_h);
      REQUIRE(coll_d.isValid());
      REQUIRE(coll_d.size() == size);

      auto div = cms::alpakatools::make_workdiv<Acc1D>(1, size);
      alpaka::exec<Acc1D>(
          queue,
          div,
          [] ALPAKA_FN_ACC(Acc1D const& acc, TestSoA::ConstView view) {
            assert(view.metadata().size() == size);
            assert(view.num() == 20);
            for (int i : cms::alpakatools::uniform_elements(acc)) {
              assert(view.id(i) == i * 2 + 1);
            }
          },
          coll_d.const_view());

      alpaka::wait(queue);
    }
#endif
  }
}
