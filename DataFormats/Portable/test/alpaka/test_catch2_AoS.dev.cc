#include <algorithm>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using Catch::Matchers::WithinAbs;

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

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

GENERATE_SOA_LAYOUT(SoAScalarsTemplate, SOA_SCALAR(int, id), SOA_SCALAR(int, type), SOA_SCALAR(float, energy))

GENERATE_SOA_LAYOUT(
    SimpleLayoutTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z), SOA_COLUMN(float, t))

GENERATE_SOA_BLOCKS(SoABlocksTemplate,
                    SOA_BLOCK(position, SoAPositionTemplate),
                    SOA_BLOCK(pca, SoAPCATemplate),
                    SOA_BLOCK(scalars, SoAScalarsTemplate))

GENERATE_SOA_BLOCKS(NestedBlocksTemplate, SOA_BLOCK(blocks, SoABlocksTemplate), SOA_BLOCK(simple, SimpleLayoutTemplate))

using NestedBlocks = NestedBlocksTemplate<>;
using NestedAoSBlocks = NestedBlocks::AoSWrapper;
using NestedBlocksView = NestedBlocks::View;
using NestedBlocksConstView = NestedBlocks::ConstView;

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

struct FillNestedBlocks {
  template <typename TAcc, typename SoAView>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, SoAView view) const {
    if (cms::alpakatools::once_per_grid(acc)) {
      view.blocks().position().detectorType() = 1;
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[0])) {
      view.blocks().position()[i] = {0.1f, 0.2f, 0.3f};
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[1])) {
      view.blocks().pca()[i].vector_1() = 0.0f;
      view.blocks().pca()[i].vector_2() = 0.0f;
      view.blocks().pca()[i].vector_3() = 1.0f;
      view.blocks().pca()[i].candidateDirection() = Eigen::Vector3d(1.0, 0.0, 0.0);
    }
    if (cms::alpakatools::once_per_grid(acc)) {
      view.blocks().scalars().id() = 42;
      view.blocks().scalars().type() = 1;
      view.blocks().scalars().energy() = 100.0f;
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[3])) {
      view.simple()[i] = {2.1f, 2.2f, 2.3f, 2.4f};
    }
  }
};

template <typename ConstView>
void verifyHostView(ConstView const& view, float eps = 1.e-6f) {
  const float fElems = static_cast<float>(view.metadata().size());

  for (auto i = 0; i < view.metadata().size(); ++i) {
    const float fi = static_cast<float>(i);
    const auto& elem = view[i];

    REQUIRE_THAT(elem.x(), WithinAbs(fi + 0.0f * fElems, eps));
    REQUIRE_THAT(elem.y(), WithinAbs(fi + 1.0f * fElems, eps));
    REQUIRE_THAT(elem.z(), WithinAbs(fi + 2.0f * fElems, eps));

    REQUIRE_THAT(elem.exampleVector()(0), WithinAbs(fi + 3.0f * fElems, eps));
    REQUIRE_THAT(elem.exampleVector()(1), WithinAbs(fi + 4.0f * fElems, eps));
    REQUIRE_THAT(elem.exampleVector()(2), WithinAbs(fi + 5.0f * fElems, eps));
  }

  REQUIRE(view.s1() == 1);
  REQUIRE_THAT(view.s3(), WithinAbs(3.0f, eps));
  REQUIRE_THAT(view.s2(), WithinAbs(2.0f, eps));
}

template <typename ConstView>
void verifyNestedHostView(ConstView const& view, float eps = 1.e-6f) {
  REQUIRE(view.blocks().position().metadata().size() == 11);
  REQUIRE(view.blocks().pca().metadata().size() == 12);
  REQUIRE(view.blocks().scalars().metadata().size() == 13);
  REQUIRE(view.simple().metadata().size() == 14);

  REQUIRE(view.blocks().position().detectorType() == 1);
  for (int i = 0; i < view.metadata().size()[0]; ++i) {
    const auto pos = view.blocks().position()[i];
    REQUIRE_THAT(pos.x(), WithinAbs(0.1f, eps));
    REQUIRE_THAT(pos.y(), WithinAbs(0.2f, eps));
    REQUIRE_THAT(pos.z(), WithinAbs(0.3f, eps));
  }

  for (int i = 0; i < view.metadata().size()[1]; ++i) {
    const auto pca = view.blocks().pca()[i];
    REQUIRE_THAT(pca.vector_1(), WithinAbs(0.0f, eps));
    REQUIRE_THAT(pca.vector_2(), WithinAbs(0.0f, eps));
    REQUIRE_THAT(pca.vector_3(), WithinAbs(1.0f, eps));
    REQUIRE_THAT(pca.candidateDirection()[0], WithinAbs(1.0, static_cast<double>(eps)));
    REQUIRE_THAT(pca.candidateDirection()[1], WithinAbs(0.0, static_cast<double>(eps)));
    REQUIRE_THAT(pca.candidateDirection()[2], WithinAbs(0.0, static_cast<double>(eps)));
  }

  REQUIRE(view.blocks().scalars().id() == 42);
  REQUIRE(view.blocks().scalars().type() == 1);
  REQUIRE_THAT(view.blocks().scalars().energy(), WithinAbs(100.0f, eps));

  for (int i = 0; i < view.metadata().size()[3]; ++i) {
    const auto simple = view.simple()[i];
    REQUIRE_THAT(simple.x(), WithinAbs(2.1f, eps));
    REQUIRE_THAT(simple.y(), WithinAbs(2.2f, eps));
    REQUIRE_THAT(simple.z(), WithinAbs(2.3f, eps));
    REQUIRE_THAT(simple.t(), WithinAbs(2.4f, eps));
  }
}

TEST_CASE("AoS testcase for PortableCollection", "[PortableCollectionAOS]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
        "the test will be skipped.");
  }

  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;

    Queue queue(device);

    SECTION("Check with simple layout") {
      // number of elements for this test case
      const std::size_t elems = 10;

      // Portable Collections using SoA layout
      PortableCollection<Device, SoA> soaCollection(queue, elems);
      SoAView& soaCollectionView = soaCollection.view();

      PortableCollection<Device, AoS> aosCollection(queue, elems);

      auto blockSize = 64;
      auto numberOfBlocks = cms::alpakatools::divide_up_by(elems, blockSize);
      const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue, workDiv, FillSoA{}, soaCollectionView);
      alpaka::wait(queue);

      // Transpose the data from SoA to AoS
      aosCollection.transpose<Acc1D>(queue, soaCollection);
      alpaka::wait(queue);

      // Check the results on host
      PortableHostCollection<AoS> aosCollectionHost(queue, elems);
      const AoSConstView& aosCollectionHostView = aosCollectionHost.const_view();

      alpaka::memcpy(queue, aosCollectionHost.buffer(), aosCollection.buffer());
      alpaka::wait(queue);

      verifyHostView(aosCollectionHostView);

      // Transpose the data back from AoS to SoA
      PortableCollection<Device, SoA> soaCollection2(queue, elems);

      soaCollection2.transpose<Acc1D>(queue, aosCollection);
      alpaka::wait(queue);

      // Check the results on host
      PortableHostCollection<SoA> soaCollectionHost2(queue, elems);
      const SoAConstView& soaCollectionHost2View = soaCollectionHost2.const_view();

      alpaka::memcpy(queue, soaCollectionHost2.buffer(), soaCollection2.buffer());
      alpaka::wait(queue);

      verifyHostView(soaCollectionHost2View);
    }

    SECTION("Check nested blocks layout") {
      // number of elements for this test case
      std::array<cms::soa::size_type, 4> elems{{11, 12, 13, 14}};

      // Portable Collections using SoA layout
      PortableCollection<Device, NestedBlocks> soaCollection(queue, elems);
      NestedBlocksView& soaCollectionView = soaCollection.view();

      PortableCollection<Device, NestedAoSBlocks> aosCollection(queue, elems);

      auto blockSize = 64;
      auto numberOfBlocks = cms::alpakatools::divide_up_by(std::ranges::max(elems), blockSize);
      const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue, workDiv, FillNestedBlocks{}, soaCollectionView);
      alpaka::wait(queue);

      // Transpose the data from SoA to AoS
      aosCollection.transpose<Acc1D>(queue, soaCollection);
      alpaka::wait(queue);

      // Check the results on host
      PortableHostCollection<NestedAoSBlocks> aosCollectionHost(queue, elems);
      const NestedAoSBlocks::ConstView& aosCollectionHostView = aosCollectionHost.const_view();

      alpaka::memcpy(queue, aosCollectionHost.buffer(), aosCollection.buffer());
      alpaka::wait(queue);

      verifyNestedHostView(aosCollectionHostView);

      // Transpose the data back from AoS to SoA
      PortableCollection<Device, NestedBlocks> soaCollection2(queue, elems);

      soaCollection2.transpose<Acc1D>(queue, aosCollection);
      alpaka::wait(queue);

      // Check the results on host
      PortableHostCollection<NestedBlocks> soaCollectionHost2(queue, elems);
      const NestedBlocksConstView& soaCollectionHost2View = soaCollectionHost2.const_view();

      alpaka::memcpy(queue, soaCollectionHost2.buffer(), soaCollection2.buffer());
      alpaka::wait(queue);

      verifyNestedHostView(soaCollectionHost2View);
    }
  }
}
