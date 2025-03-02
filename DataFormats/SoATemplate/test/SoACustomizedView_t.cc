#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

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

GENERATE_SOA_LAYOUT(CustomizedSoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using CustomizedSoA = CustomizedSoATemplate<>;
using CustomizedSoAView = CustomizedSoA::View;
using CustomizedSoAConstView = CustomizedSoA::ConstView;

TEST_CASE("SoACustomizedView") {
  // common number of elements for the SoAs
  const std::size_t elems = 10;

  // buffer sizes
  const std::size_t positionBufferSize = SoAPosition::computeDataSize(elems);
  const std::size_t pcaBufferSize = SoAPCA::computeDataSize(elems);

  // memory buffer for the SoA of positions
  std::unique_ptr<std::byte, decltype(std::free) *> bufferPos{
      reinterpret_cast<std::byte *>(aligned_alloc(SoAPosition::alignment, positionBufferSize)), std::free};
  // memory buffer for the SoA of the PCA
  std::unique_ptr<std::byte, decltype(std::free) *> bufferPCA{
      reinterpret_cast<std::byte *>(aligned_alloc(SoAPCA::alignment, pcaBufferSize)), std::free};

  // SoA Layouts
  SoAPosition position{bufferPos.get(), elems};
  SoAPCA pca{bufferPCA.get(), elems};

  // SoA Views
  SoAPositionView positionView{position};
  SoAPositionConstView positionConstView{position};
  SoAPCAView pcaView{pca};
  SoAPCAConstView pcaConstView{pca};

  // fill up
  for (size_t i = 0; i < elems; i++) {
    positionView.x()[i] = static_cast<float>(i);
    positionView.y()[i] = static_cast<float>(i) * 2.0f;
    positionView.z()[i] = static_cast<float>(i) * 3.0f;
  }
  positionView.detectorType() = 1;

  float time = 0.01;
  for (size_t i = 0; i < elems; i++) {
    pcaView.eigenvector_1()[i] = positionView.x()[i] / time;
    pcaView.eigenvector_2()[i] = positionView.y()[i] / time;
    pcaView.eigenvector_3()[i] = positionView.z()[i] / time;
    pcaView[i].candidateDirection()(0) = positionView[i].x() / time;
    pcaView[i].candidateDirection()(1) = positionView[i].y() / time;
    pcaView[i].candidateDirection()(2) = positionView[i].z() / time;
  }

  SECTION("Customized View") {
    // addresses and size of the SoA columns
    const auto posRecs = positionView.records();
    const auto pcaRecs = pcaView.records();

    // building the View with runtime check for the size
    CustomizedSoAView customizedView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for equality of memory addresses
    REQUIRE(customizedView.metadata().addressOf_x() == positionView.metadata().addressOf_x());
    REQUIRE(customizedView.metadata().addressOf_y() == positionView.metadata().addressOf_y());
    REQUIRE(customizedView.metadata().addressOf_z() == positionView.metadata().addressOf_z());
    REQUIRE(customizedView.metadata().addressOf_candidateDirection() ==
            pcaView.metadata().addressOf_candidateDirection());

    // Check for reference to original SoA
    customizedView.x()[3] = 0.;
    REQUIRE(customizedView.x()[3] == positionConstView.x()[3]);
  }

  SECTION("Customized ConstView") {
    // addresses and size of the SoA columns
    const auto posRecs = positionConstView.records();
    const auto pcaRecs = pcaConstView.records();

    // building the ConstView with runtime check for the size
    CustomizedSoAConstView customizedConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for equality of memory addresses
    REQUIRE(customizedConstView.metadata().addressOf_x() == positionConstView.metadata().addressOf_x());
    REQUIRE(customizedConstView.metadata().addressOf_y() == positionConstView.metadata().addressOf_y());
    REQUIRE(customizedConstView.metadata().addressOf_z() == positionConstView.metadata().addressOf_z());
    REQUIRE(customizedConstView.metadata().addressOf_candidateDirection() ==
            pcaConstView.metadata().addressOf_candidateDirection());
  }

  SECTION("Customized ConstView from Views") {
    // addresses and size of the SoA columns
    const auto posRecs = positionView.records();
    const auto pcaRecs = pcaView.records();

    // building the ConstView with runtime check for the size
    CustomizedSoAConstView customizedConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for reference to the Custom SoA - it is possible to modify the ConstView by reference modifying the Views
    positionView.x()[3] = 0.;
    REQUIRE(customizedConstView.x()[3] == positionView.x()[3]);
  }

  SECTION("Aggregate the Customized View") {
    // building the Layout
    const std::size_t customBufferSize = CustomizedSoA::computeDataSize(elems);
    std::unique_ptr<std::byte, decltype(std::free) *> bufferCustom{
        reinterpret_cast<std::byte *>(aligned_alloc(CustomizedSoA::alignment, customBufferSize)), std::free};
    CustomizedSoA customSoA(bufferCustom.get(), elems);

    // building the Customized View
    const auto posRecs = positionView.records();
    const auto pcaRecs = pcaView.records();
    CustomizedSoAView customizedView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // aggregate the columns from the view with runtime check for the size
    customSoA.deep_copy(customizedView);

    // building the View of the aggregated SoA
    CustomizedSoAView customizedAggregatedView{customSoA};

    // Check for inequality of memory addresses
    REQUIRE(customizedAggregatedView.metadata().addressOf_x() != positionConstView.metadata().addressOf_x());
    REQUIRE(customizedAggregatedView.metadata().addressOf_y() != positionConstView.metadata().addressOf_y());
    REQUIRE(customizedAggregatedView.metadata().addressOf_z() != positionConstView.metadata().addressOf_z());
    REQUIRE(customizedAggregatedView.metadata().addressOf_candidateDirection() !=
            pcaConstView.metadata().addressOf_candidateDirection());

    // Check for column alignments
    REQUIRE(0 == reinterpret_cast<uintptr_t>(customizedAggregatedView.metadata().addressOf_x()) %
                     decltype(customSoA)::alignment);
    REQUIRE(0 == reinterpret_cast<uintptr_t>(customizedAggregatedView.metadata().addressOf_y()) %
                     decltype(customSoA)::alignment);
    REQUIRE(0 == reinterpret_cast<uintptr_t>(customizedAggregatedView.metadata().addressOf_z()) %
                     decltype(customSoA)::alignment);
    REQUIRE(0 == reinterpret_cast<uintptr_t>(customizedAggregatedView.metadata().addressOf_candidateDirection()) %
                     decltype(customSoA)::alignment);

    // Check for contiguity of columns
    REQUIRE(reinterpret_cast<std::byte *>(customizedAggregatedView.metadata().addressOf_x()) +
                cms::soa::alignSize(elems * sizeof(float), CustomizedSoA::alignment) ==
            reinterpret_cast<std::byte *>(customizedAggregatedView.metadata().addressOf_y()));
    REQUIRE(reinterpret_cast<std::byte *>(customizedAggregatedView.metadata().addressOf_y()) +
                cms::soa::alignSize(elems * sizeof(float), CustomizedSoA::alignment) ==
            reinterpret_cast<std::byte *>(customizedAggregatedView.metadata().addressOf_z()));
    REQUIRE(reinterpret_cast<std::byte *>(customizedAggregatedView.metadata().addressOf_z()) +
                cms::soa::alignSize(elems * sizeof(float), CustomizedSoA::alignment) ==
            reinterpret_cast<std::byte *>(customizedAggregatedView.metadata().addressOf_candidateDirection()));

    // Check for the independency of the aggregated SoA
    customizedAggregatedView.x()[3] = 0.;
    REQUIRE(customizedAggregatedView.x()[3] != positionView.x()[3]);
  }
}
