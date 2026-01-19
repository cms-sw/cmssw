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

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, eigenvalues),
                    SOA_COLUMN(float, eigenvector_1),
                    SOA_COLUMN(float, eigenvector_2),
                    SOA_COLUMN(float, eigenvector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;

GENERATE_SOA_LAYOUT(GenericSoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using GenericSoA = GenericSoATemplate<cms::soa::CacheLineSize::IntelCPU>;
using GenericSoAView = GenericSoA::View;
using GenericSoAConstView = GenericSoA::ConstView;

// Kernel for filling the SoA
struct FillSoA {
  template <alpaka::concepts::Acc TAcc, typename PositionView, typename PCAView>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PositionView positionView, PCAView pcaView) const {
    constexpr float interval = 0.01f;
    if (cms::alpakatools::once_per_grid(acc))
      positionView.detectorType() = 1;

    for (auto local_idx : cms::alpakatools::uniform_elements(acc, positionView.metadata().size())) {
      positionView[local_idx].x() = static_cast<float>(local_idx);
      positionView[local_idx].y() = static_cast<float>(local_idx) * 2.0f;
      positionView[local_idx].z() = static_cast<float>(local_idx) * 3.0f;

      pcaView[local_idx].eigenvector_1() = positionView[local_idx].x() / interval;
      pcaView[local_idx].eigenvector_2() = positionView[local_idx].y() / interval;
      pcaView[local_idx].eigenvector_3() = positionView[local_idx].z() / interval;
      pcaView[local_idx].candidateDirection()(0) = positionView[local_idx].x() / interval;
      pcaView[local_idx].candidateDirection()(1) = positionView[local_idx].y() / interval;
      pcaView[local_idx].candidateDirection()(2) = positionView[local_idx].z() / interval;
    }
  }
};

TEST_CASE("Deep copy from SoA Generic View") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
        "the test will be skipped.");
  }

  auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);

  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;

    Queue queue(device);

    // common number of elements for the SoAs
    const std::size_t elems = 10;

    // Portable Collections
    PortableCollection<SoAPosition, Device> positionCollection(elems, queue);
    PortableCollection<SoAPCA, Device> pcaCollection(elems, queue);

    // Portable Collection Views
    SoAPositionView& positionCollectionView = positionCollection.view();
    SoAPCAView& pcaCollectionView = pcaCollection.view();
    // Portable Collection ConstViews
    const SoAPositionConstView& positionCollectionConstView = positionCollection.const_view();
    const SoAPCAConstView& pcaCollectionConstView = pcaCollection.const_view();

    // fill up
    auto blockSize = 64;
    auto numberOfBlocks = cms::alpakatools::divide_up_by(elems, blockSize);

    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDiv, FillSoA{}, positionCollectionView, pcaCollectionView);

    alpaka::wait(queue);

    SECTION("Deep copy the View host to host and device to device") {
      // addresses and size of the SoA columns
      const auto posRecs = positionCollectionView.records();
      const auto pcaRecs = pcaCollectionView.records();

      // building the View with runtime check for the size
      GenericSoAView genericView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

      // Check for equality of memory addresses
      REQUIRE(genericView.metadata().addressOf_x() == positionCollectionView.metadata().addressOf_x());
      REQUIRE(genericView.metadata().addressOf_y() == positionCollectionView.metadata().addressOf_y());
      REQUIRE(genericView.metadata().addressOf_z() == positionCollectionView.metadata().addressOf_z());
      REQUIRE(genericView.metadata().addressOf_candidateDirection() ==
              pcaCollectionView.metadata().addressOf_candidateDirection());

      // PortableCollection that will host the aggregated columns
      PortableCollection<GenericSoA, Device> genericCollection(elems, queue);
      genericCollection.deepCopy(genericView, queue);

      // Check for inequality of memory addresses[`View` section](../../DataFormats/SoATemplate/README.md#view)
      REQUIRE(genericCollection.view().metadata().addressOf_x() != positionCollectionView.metadata().addressOf_x());
      REQUIRE(genericCollection.view().metadata().addressOf_y() != positionCollectionView.metadata().addressOf_y());
      REQUIRE(genericCollection.view().metadata().addressOf_z() != positionCollectionView.metadata().addressOf_z());
      REQUIRE(genericCollection.view().metadata().addressOf_candidateDirection() !=
              pcaCollectionView.metadata().addressOf_candidateDirection());
    }

    SECTION("Deep copy the ConstView host to host and device to device") {
      // addresses and size of the SoA columns
      const auto posRecs = positionCollectionConstView.records();
      const auto pcaRecs = pcaCollectionConstView.records();

      // building the View with runtime check for the size
      GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

      // Check for equality of memory addresses
      REQUIRE(genericConstView.metadata().addressOf_x() == positionCollectionView.metadata().addressOf_x());
      REQUIRE(genericConstView.metadata().addressOf_y() == positionCollectionView.metadata().addressOf_y());
      REQUIRE(genericConstView.metadata().addressOf_z() == positionCollectionView.metadata().addressOf_z());
      REQUIRE(genericConstView.metadata().addressOf_candidateDirection() ==
              pcaCollectionView.metadata().addressOf_candidateDirection());

      // PortableCollection that will host the aggregated columns
      PortableCollection<GenericSoA, Device> genericCollection(elems, queue);
      genericCollection.deepCopy(genericConstView, queue);

      // Check for inequality of memory addresses
      REQUIRE(genericCollection.view().metadata().addressOf_x() != positionCollectionView.metadata().addressOf_x());
      REQUIRE(genericCollection.view().metadata().addressOf_y() != positionCollectionView.metadata().addressOf_y());
      REQUIRE(genericCollection.view().metadata().addressOf_z() != positionCollectionView.metadata().addressOf_z());
      REQUIRE(genericCollection.view().metadata().addressOf_candidateDirection() !=
              pcaCollectionView.metadata().addressOf_candidateDirection());

      // Check for correctness of the copy
      PortableHostCollection<GenericSoA> genericHostCollection(elems, queue);
      PortableHostCollection<SoAPosition> positionHostCollection(elems, queue);
      PortableHostCollection<SoAPCA> pcaHostCollection(elems, queue);

      alpaka::memcpy(queue, genericHostCollection.buffer(), genericCollection.buffer());
      alpaka::memcpy(queue, positionHostCollection.buffer(), positionCollection.buffer());
      alpaka::memcpy(queue, pcaHostCollection.buffer(), pcaCollection.buffer());

      alpaka::wait(queue);

      const GenericSoAConstView& genericViewHostCollection = genericHostCollection.const_view();
      const SoAPositionConstView& positionViewHostCollection = positionHostCollection.const_view();
      const SoAPCAConstView& pcaViewHostCollection = pcaHostCollection.const_view();

      for (size_t i = 0; i < elems; i++) {
        REQUIRE(genericViewHostCollection[i].x() == positionViewHostCollection[i].x());
        REQUIRE(genericViewHostCollection[i].y() == positionViewHostCollection[i].y());
        REQUIRE(genericViewHostCollection[i].z() == positionViewHostCollection[i].z());
        REQUIRE(genericViewHostCollection[i].candidateDirection()(0) ==
                pcaViewHostCollection[i].candidateDirection()(0));
        REQUIRE(genericViewHostCollection[i].candidateDirection()(1) ==
                pcaViewHostCollection[i].candidateDirection()(1));
        REQUIRE(genericViewHostCollection[i].candidateDirection()(2) ==
                pcaViewHostCollection[i].candidateDirection()(2));
      }
    }

    SECTION("Deep copy the ConstView device to host") {
      // addresses and size of the SoA columns
      const auto posRecs = positionCollectionConstView.records();
      const auto pcaRecs = pcaCollectionConstView.records();

      // building the View with runtime check for the size
      GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

      // Check for equality of memory addresses
      REQUIRE(genericConstView.metadata().addressOf_x() == positionCollectionView.metadata().addressOf_x());
      REQUIRE(genericConstView.metadata().addressOf_y() == positionCollectionView.metadata().addressOf_y());
      REQUIRE(genericConstView.metadata().addressOf_z() == positionCollectionView.metadata().addressOf_z());
      REQUIRE(genericConstView.metadata().addressOf_candidateDirection() ==
              pcaCollectionView.metadata().addressOf_candidateDirection());

      // PortableCollection that will host the aggregated columns
      PortableHostCollection<GenericSoA> genericCollection(elems, queue);
      genericCollection.deepCopy(genericConstView, queue);

      // Check for inequality of memory addresses
      REQUIRE(genericCollection.view().metadata().addressOf_x() != positionCollectionView.metadata().addressOf_x());
      REQUIRE(genericCollection.view().metadata().addressOf_y() != positionCollectionView.metadata().addressOf_y());
      REQUIRE(genericCollection.view().metadata().addressOf_z() != positionCollectionView.metadata().addressOf_z());
      REQUIRE(genericCollection.view().metadata().addressOf_candidateDirection() !=
              pcaCollectionView.metadata().addressOf_candidateDirection());

      // Check for correctness of the copy
      PortableHostCollection<SoAPosition> positionHostCollection(elems, queue);
      PortableHostCollection<SoAPCA> pcaHostCollection(elems, queue);

      alpaka::memcpy(queue, positionHostCollection.buffer(), positionCollection.buffer());
      alpaka::memcpy(queue, pcaHostCollection.buffer(), pcaCollection.buffer());

      alpaka::wait(queue);

      const GenericSoAConstView& genericViewCollection = genericCollection.const_view();
      const SoAPositionConstView& positionViewHostCollection = positionHostCollection.const_view();
      const SoAPCAConstView& pcaViewHostCollection = pcaHostCollection.const_view();

      for (size_t i = 0; i < elems; i++) {
        REQUIRE(genericViewCollection[i].x() == positionViewHostCollection[i].x());
        REQUIRE(genericViewCollection[i].y() == positionViewHostCollection[i].y());
        REQUIRE(genericViewCollection[i].z() == positionViewHostCollection[i].z());
        REQUIRE(genericViewCollection[i].candidateDirection()(0) == pcaViewHostCollection[i].candidateDirection()(0));
        REQUIRE(genericViewCollection[i].candidateDirection()(1) == pcaViewHostCollection[i].candidateDirection()(1));
        REQUIRE(genericViewCollection[i].candidateDirection()(2) == pcaViewHostCollection[i].candidateDirection()(2));
      }
    }

    SECTION("Deep copy the ConstView host to device") {
      PortableHostCollection<SoAPosition> positionHostCollection(elems, queue);
      PortableHostCollection<SoAPCA> pcaHostCollection(elems, queue);

      alpaka::memcpy(queue, positionHostCollection.buffer(), positionCollection.buffer());
      alpaka::memcpy(queue, pcaHostCollection.buffer(), pcaCollection.buffer());

      const SoAPositionConstView& positionViewHostCollection = positionHostCollection.const_view();
      const SoAPCAConstView& pcaViewHostCollection = pcaHostCollection.const_view();

      // addresses and size of the SoA columns
      const auto posRecs = positionViewHostCollection.records();
      const auto pcaRecs = pcaViewHostCollection.records();

      // building the View with runtime check for the size
      GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

      // Check for equality of memory addresses
      REQUIRE(genericConstView.metadata().addressOf_x() == positionViewHostCollection.metadata().addressOf_x());
      REQUIRE(genericConstView.metadata().addressOf_y() == positionViewHostCollection.metadata().addressOf_y());
      REQUIRE(genericConstView.metadata().addressOf_z() == positionViewHostCollection.metadata().addressOf_z());
      REQUIRE(genericConstView.metadata().addressOf_candidateDirection() ==
              pcaViewHostCollection.metadata().addressOf_candidateDirection());

      // PortableCollection that will host the aggregated columns
      PortableCollection<GenericSoA, Device> genericCollection(elems, queue);
      genericCollection.deepCopy(genericConstView, queue);

      // Check for inequality of memory addresses
      REQUIRE(genericCollection.view().metadata().addressOf_x() != positionViewHostCollection.metadata().addressOf_x());
      REQUIRE(genericCollection.view().metadata().addressOf_y() != positionViewHostCollection.metadata().addressOf_y());
      REQUIRE(genericCollection.view().metadata().addressOf_z() != positionViewHostCollection.metadata().addressOf_z());
      REQUIRE(genericCollection.view().metadata().addressOf_candidateDirection() !=
              pcaViewHostCollection.metadata().addressOf_candidateDirection());

      // Check for correctness of the copy
      PortableHostCollection<GenericSoA> genericHostCollection(elems, queue);

      alpaka::memcpy(queue, genericHostCollection.buffer(), genericCollection.buffer());

      alpaka::wait(queue);

      const GenericSoAConstView& genericViewHostCollection = genericHostCollection.const_view();

      for (size_t i = 0; i < elems; i++) {
        REQUIRE(genericViewHostCollection[i].x() == positionViewHostCollection[i].x());
        REQUIRE(genericViewHostCollection[i].y() == positionViewHostCollection[i].y());
        REQUIRE(genericViewHostCollection[i].z() == positionViewHostCollection[i].z());
        REQUIRE(genericViewHostCollection[i].candidateDirection()(0) ==
                pcaViewHostCollection[i].candidateDirection()(0));
        REQUIRE(genericViewHostCollection[i].candidateDirection()(1) ==
                pcaViewHostCollection[i].candidateDirection()(1));
        REQUIRE(genericViewHostCollection[i].candidateDirection()(2) ==
                pcaViewHostCollection[i].candidateDirection()(2));
      }
    }
  }
}
