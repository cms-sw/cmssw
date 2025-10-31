#include <Eigen/Core>
#include <Eigen/Dense>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_SCALAR(int, s1),
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(float, s2),
                    SOA_EIGEN_COLUMN(Eigen::Vector3f, exampleVector),
                    SOA_SCALAR(float, s3))

using SoA = SoATemplate<>;
using AoS = SoA::AoSWrapper;
using SoAView = SoA::View;
using AoSView = AoS::View;
using SoAConstView = SoA::ConstView;
using AoSConstView = AoS::ConstView;

struct FillSoA {
  template <typename TAcc, typename SoAView>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, SoAView view) const {
    if (cms::alpakatools::once_per_grid(acc)) {
      view.s1() = 1;
      view.s2() = 2.0f;
      view.s3() = 3.0f;
    }

    const float n = static_cast<float>(view.metadata().size());

    for (auto local_idx : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
      view[local_idx].x() = static_cast<float>(local_idx) + 0.0f * n;
      view[local_idx].y() = static_cast<float>(local_idx) + 1.0f * n;
      view[local_idx].z() = static_cast<float>(local_idx) + 2.0f * n;

      view[local_idx].exampleVector()(0) = static_cast<float>(local_idx) + 3.0f * n;
      view[local_idx].exampleVector()(1) = static_cast<float>(local_idx) + 4.0f * n;
      view[local_idx].exampleVector()(2) = static_cast<float>(local_idx) + 5.0f * n;
    }
  }
};

TEST_CASE("AoS testcase for PortableCollection", "[PortableCollectionAOS]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
        "the test will be skipped.");
  }

  auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);

  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;

    Queue queue(device);

    // number of elements for this test case
    const std::size_t elems = 10;

    // Portable Collections using SoA layout
    PortableCollection<SoA, Device> soaCollection(elems, queue);
    SoAView& soaCollectionView = soaCollection.view();

    PortableCollection<AoS, Device> aosCollection(elems, queue);

    auto blockSize = 64;
    auto numberOfBlocks = cms::alpakatools::divide_up_by(elems, blockSize);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDiv, FillSoA{}, soaCollectionView);
    alpaka::wait(queue);

    // Transpose the data from SoA to AoS
    aosCollection.transpose<Acc1D>(soaCollection, queue);
    alpaka::wait(queue);

    // Check the results on host
    PortableHostCollection<AoS> aosCollectionHost(elems, queue);
    const AoSConstView& aosCollectionHostView = aosCollectionHost.const_view();

    alpaka::memcpy(queue, aosCollectionHost.buffer(), aosCollection.buffer());
    alpaka::wait(queue);

    for (size_t i = 0; i < elems; i++) {
      REQUIRE_THAT(aosCollectionHostView[i].x(),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.0f * static_cast<float>(elems), 1.e-6));
      REQUIRE_THAT(aosCollectionHostView[i].y(),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 1.0f * static_cast<float>(elems), 1.e-6));
      REQUIRE_THAT(aosCollectionHostView[i].z(),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 2.0f * static_cast<float>(elems), 1.e-6));

      REQUIRE_THAT(aosCollectionHostView[i].exampleVector()(0),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 3.0f * static_cast<float>(elems), 1.e-6));
      REQUIRE_THAT(aosCollectionHostView[i].exampleVector()(1),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 4.0f * static_cast<float>(elems), 1.e-6));
      REQUIRE_THAT(aosCollectionHostView[i].exampleVector()(2),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 5.0f * static_cast<float>(elems), 1.e-6));
    }
    REQUIRE(aosCollectionHostView.s1() == 1);
    REQUIRE_THAT(aosCollectionHostView.s2(), Catch::Matchers::WithinAbs(2.0f, 1.e-6));
    REQUIRE_THAT(aosCollectionHostView.s3(), Catch::Matchers::WithinAbs(3.0f, 1.e-6));

    // Transpose the data back from AoS to SoA
    PortableCollection<SoA, Device> soaCollection2(elems, queue);

    soaCollection2.transpose<Acc1D>(aosCollection, queue);
    alpaka::wait(queue);

    // Check the results on host
    PortableHostCollection<SoA> soaCollectionHost2(elems, queue);
    const SoAConstView& soaCollectionHost2View = soaCollectionHost2.const_view();

    alpaka::memcpy(queue, soaCollectionHost2.buffer(), soaCollection2.buffer());
    alpaka::wait(queue);

    for (size_t i = 0; i < elems; i++) {
      REQUIRE_THAT(soaCollectionHost2View[i].x(),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.0f * static_cast<float>(elems), 1.e-6));
      REQUIRE_THAT(soaCollectionHost2View[i].y(),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 1.0f * static_cast<float>(elems), 1.e-6));
      REQUIRE_THAT(soaCollectionHost2View[i].z(),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 2.0f * static_cast<float>(elems), 1.e-6));

      REQUIRE_THAT(soaCollectionHost2View[i].exampleVector()(0),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 3.0f * static_cast<float>(elems), 1.e-6));
      REQUIRE_THAT(soaCollectionHost2View[i].exampleVector()(1),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 4.0f * static_cast<float>(elems), 1.e-6));
      REQUIRE_THAT(soaCollectionHost2View[i].exampleVector()(2),
                   Catch::Matchers::WithinAbs(static_cast<float>(i) + 5.0f * static_cast<float>(elems), 1.e-6));
    }
    REQUIRE(soaCollectionHost2View.s1() == 1);
    REQUIRE_THAT(soaCollectionHost2View.s2(), Catch::Matchers::WithinAbs(2.0f, 1.e-6));
    REQUIRE_THAT(soaCollectionHost2View.s3(), Catch::Matchers::WithinAbs(3.0f, 1.e-6));
    REQUIRE_THAT(soaCollectionHost2View.s3(), Catch::Matchers::WithinAbs(3.0f, 1.e-6));
  }
}
