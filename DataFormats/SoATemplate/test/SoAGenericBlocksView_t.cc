#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"

// This file tests the possibility to build a SoABlocks View/ConstView starting from
// existing Views/ConstViews of other SoAs. It is also possible then to deepCopy these
// Views/ConstViews into a SoABlocks instance. This can be considered as the expansion of
// the SoAGenericView_t.cc test to the case of blocks.
// N.B. It is also possible to build a SoAGenericBlocksView/ConstView starting from
// a SoAGenericView, but this behaviour is not tested here.

constexpr float step = 0.01;

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;

GENERATE_SOA_BLOCKS(SoAGenericBlocksTemplate, SOA_BLOCK(position, SoAPositionTemplate), SOA_BLOCK(pca, SoAPCATemplate))

using SoAGenericBlocks = SoAGenericBlocksTemplate<>;
using SoAGenericBlocksView = SoAGenericBlocks::View;
using SoAGenericBlocksConstView = SoAGenericBlocks::ConstView;

TEST_CASE("SoAGenericBlocks") {
  // different number of elements for the SoAs
  const std::size_t elemsPos = 10;
  const std::size_t elemsPCA = 20;

  // buffer sizes
  const std::size_t positionBufferSize = SoAPosition::computeDataSize(elemsPos);
  const std::size_t pcaBufferSize = SoAPCA::computeDataSize(elemsPCA);

  // memory buffer for the SoA of positions
  std::unique_ptr<std::byte, decltype(std::free) *> bufferPos{
      reinterpret_cast<std::byte *>(aligned_alloc(SoAPosition::alignment, positionBufferSize)), std::free};
  // memory buffer for the SoA of the PCA
  std::unique_ptr<std::byte, decltype(std::free) *> bufferPCA{
      reinterpret_cast<std::byte *>(aligned_alloc(SoAPCA::alignment, pcaBufferSize)), std::free};

  // SoA Layouts
  SoAPosition position{bufferPos.get(), elemsPos};
  SoAPCA pca{bufferPCA.get(), elemsPCA};

  // SoA Views
  SoAPositionView positionView{position};
  SoAPositionConstView positionConstView{position};
  SoAPCAView pcaView{pca};
  SoAPCAConstView pcaConstView{pca};

  // fill up
  for (size_t i = 0; i < elemsPos; i++) {
    positionView[i] = {i * 1.0f, i * 2.0f, i * 3.0f};
  }
  positionView.detectorType() = 1;

  for (size_t i = 0; i < elemsPCA; i++) {
    pcaView[i].vector_1() = (i * 1.0f) / step;
    pcaView[i].vector_2() = (i * 2.0f) / step;
    pcaView[i].vector_3() = (i * 3.0f) / step;
    pcaView[i].candidateDirection()(0) = (i * 1.0) / step;
    pcaView[i].candidateDirection()(1) = (i * 2.0) / step;
    pcaView[i].candidateDirection()(2) = (i * 3.0) / step;
  }

  SECTION("GenericBlocks View") {
    // building the SoABlocks View, there is no need for runtime check for the size since they are different
    SoAGenericBlocksView genericBlocksView{positionView, pcaView};

    // Verify metadata
    REQUIRE(genericBlocksView.metadata().size()[0] == elemsPos);
    REQUIRE(genericBlocksView.position().metadata().size() == elemsPos);
    REQUIRE(genericBlocksView.metadata().size()[1] == elemsPCA);
    REQUIRE(genericBlocksView.pca().metadata().size() == elemsPCA);

    // Check for equality of memory addresses
    REQUIRE(genericBlocksView.position().metadata().addressOf_x() == positionView.metadata().addressOf_x());
    REQUIRE(genericBlocksView.position().metadata().addressOf_y() == positionView.metadata().addressOf_y());
    REQUIRE(genericBlocksView.position().metadata().addressOf_z() == positionView.metadata().addressOf_z());
    REQUIRE(genericBlocksView.pca().metadata().addressOf_candidateDirection() ==
            pcaView.metadata().addressOf_candidateDirection());

    // Verify data
    for (int i = 0; i < genericBlocksView.position().metadata().size(); ++i) {
      auto pos = genericBlocksView.position()[i];
      REQUIRE(pos.x() == static_cast<float>(i));
      REQUIRE(pos.y() == static_cast<float>(i * 2.0f));
      REQUIRE(pos.z() == static_cast<float>(i * 3.0f));
    }
    for (int i = 0; i < genericBlocksView.pca().metadata().size(); ++i) {
      auto pca = genericBlocksView.pca()[i];
      REQUIRE(pca.vector_1() == static_cast<float>(i) / step);
      REQUIRE(pca.vector_2() == static_cast<float>(i * 2.0f) / step);
      REQUIRE(pca.vector_3() == static_cast<float>(i * 3.0f) / step);
      REQUIRE(pca.candidateDirection()(0) == static_cast<double>(i) / step);
      REQUIRE(pca.candidateDirection()(1) == static_cast<double>(i * 2.0) / step);
      REQUIRE(pca.candidateDirection()(2) == static_cast<double>(i * 3.0) / step);
    }
  }

  SECTION("GenericBlocks ConstView") {
    // building the SoABlocks View, there is no need for runtime check for the size since they are different
    SoAGenericBlocksConstView genericBlocksConstView{positionConstView, pcaConstView};

    // Verify metadata
    REQUIRE(genericBlocksConstView.metadata().size()[0] == elemsPos);
    REQUIRE(genericBlocksConstView.position().metadata().size() == elemsPos);
    REQUIRE(genericBlocksConstView.metadata().size()[1] == elemsPCA);
    REQUIRE(genericBlocksConstView.pca().metadata().size() == elemsPCA);

    // Check for equality of memory addresses
    REQUIRE(genericBlocksConstView.position().metadata().addressOf_x() == positionConstView.metadata().addressOf_x());
    REQUIRE(genericBlocksConstView.position().metadata().addressOf_y() == positionConstView.metadata().addressOf_y());
    REQUIRE(genericBlocksConstView.position().metadata().addressOf_z() == positionConstView.metadata().addressOf_z());
    REQUIRE(genericBlocksConstView.pca().metadata().addressOf_candidateDirection() ==
            pcaConstView.metadata().addressOf_candidateDirection());

    // Verify data
    for (int i = 0; i < genericBlocksConstView.position().metadata().size(); ++i) {
      auto pos = genericBlocksConstView.position()[i];
      REQUIRE(pos.x() == static_cast<float>(i));
      REQUIRE(pos.y() == static_cast<float>(i * 2.0f));
      REQUIRE(pos.z() == static_cast<float>(i * 3.0f));
    }
    for (int i = 0; i < genericBlocksConstView.pca().metadata().size(); ++i) {
      auto pca = genericBlocksConstView.pca()[i];
      REQUIRE(pca.vector_1() == static_cast<float>(i) / step);
      REQUIRE(pca.vector_2() == static_cast<float>(i * 2.0f) / step);
      REQUIRE(pca.vector_3() == static_cast<float>(i * 3.0f) / step);
      REQUIRE(pca.candidateDirection()(0) == static_cast<double>(i) / step);
      REQUIRE(pca.candidateDirection()(1) == static_cast<double>(i * 2.0) / step);
      REQUIRE(pca.candidateDirection()(2) == static_cast<double>(i * 3.0) / step);
    }
  }

  SECTION("GenericBlocks ConstView from Views") {
    // building the SoABlocks ConstView, there is no need for runtime check for the size since they are different
    SoAGenericBlocksConstView genericBlocksConstView{positionView, pcaView};

    // Verify data
    for (int i = 0; i < genericBlocksConstView.position().metadata().size(); ++i) {
      auto pos = genericBlocksConstView.position()[i];
      REQUIRE(pos.x() == static_cast<float>(i));
      REQUIRE(pos.y() == static_cast<float>(i * 2.0f));
      REQUIRE(pos.z() == static_cast<float>(i * 3.0f));
    }
    for (int i = 0; i < genericBlocksConstView.pca().metadata().size(); ++i) {
      auto pca = genericBlocksConstView.pca()[i];
      REQUIRE(pca.vector_1() == static_cast<float>(i) / step);
      REQUIRE(pca.vector_2() == static_cast<float>(i * 2.0f) / step);
      REQUIRE(pca.vector_3() == static_cast<float>(i * 3.0f) / step);
      REQUIRE(pca.candidateDirection()(0) == static_cast<double>(i) / step);
      REQUIRE(pca.candidateDirection()(1) == static_cast<double>(i * 2.0) / step);
      REQUIRE(pca.candidateDirection()(2) == static_cast<double>(i * 3.0) / step);
    }

    // Check for reference to original SoA - it is possible to modify the ConstView by reference modifying the Views
    positionView[3].x() = 0.;
    REQUIRE(genericBlocksConstView.position()[3].x() == positionView[3].x());
  }

  SECTION("Deep copy the GeenericBlocks View") {
    // building the SoABlocks View, there is no need for runtime check for the size since they are different
    SoAGenericBlocksView genericBlocksView{positionView, pcaView};

    // Instantiate a SoABlocks
    std::array<cms::soa::size_type, 2> sizes{{elemsPos, elemsPCA}};
    const std::size_t blocksBufferSize = SoAGenericBlocks::computeDataSize(sizes);
    std::unique_ptr<std::byte, decltype(std::free) *> bufferBlocks{
        reinterpret_cast<std::byte *>(aligned_alloc(SoAGenericBlocks::alignment, blocksBufferSize)), std::free};

    SoAGenericBlocks genericBlocks{bufferBlocks.get(), sizes};

    genericBlocks.deepCopy(genericBlocksView);

    SoAGenericBlocksView genericSoABlocksView{genericBlocks};
    // Check for inequality of memory addresses
    REQUIRE(genericSoABlocksView.position().metadata().addressOf_x() != positionConstView.metadata().addressOf_x());
    REQUIRE(genericSoABlocksView.position().metadata().addressOf_y() != positionConstView.metadata().addressOf_y());
    REQUIRE(genericSoABlocksView.position().metadata().addressOf_z() != positionConstView.metadata().addressOf_z());
    REQUIRE(genericSoABlocksView.pca().metadata().addressOf_candidateDirection() !=
            pcaConstView.metadata().addressOf_candidateDirection());

    // Check for contiguity of columns
    REQUIRE(reinterpret_cast<std::byte *>(genericSoABlocksView.position().metadata().addressOf_x()) +
                cms::soa::alignSize(elemsPos * sizeof(float), SoAGenericBlocks::alignment) ==
            reinterpret_cast<std::byte *>(genericSoABlocksView.position().metadata().addressOf_y()));
    REQUIRE(reinterpret_cast<std::byte *>(genericSoABlocksView.position().metadata().addressOf_y()) +
                cms::soa::alignSize(elemsPos * sizeof(float), SoAGenericBlocks::alignment) ==
            reinterpret_cast<std::byte *>(genericSoABlocksView.position().metadata().addressOf_z()));
    REQUIRE(reinterpret_cast<std::byte *>(genericSoABlocksView.position().metadata().addressOf_z()) +
                cms::soa::alignSize(elemsPos * sizeof(float), SoAGenericBlocks::alignment) ==
            reinterpret_cast<std::byte *>(genericSoABlocksView.position().metadata().addressOf_detectorType()));
    REQUIRE(reinterpret_cast<std::byte *>(genericSoABlocksView.position().metadata().addressOf_detectorType()) +
                cms::soa::alignSize(sizeof(int), SoAGenericBlocks::alignment) ==
            reinterpret_cast<std::byte *>(genericSoABlocksView.pca().metadata().addressOf_vector_1()));
    REQUIRE(reinterpret_cast<std::byte *>(genericSoABlocksView.pca().metadata().addressOf_vector_1()) +
                cms::soa::alignSize(elemsPCA * sizeof(float), SoAGenericBlocks::alignment) ==
            reinterpret_cast<std::byte *>(genericSoABlocksView.pca().metadata().addressOf_vector_2()));
    REQUIRE(reinterpret_cast<std::byte *>(genericSoABlocksView.pca().metadata().addressOf_vector_2()) +
                cms::soa::alignSize(elemsPCA * sizeof(float), SoAGenericBlocks::alignment) ==
            reinterpret_cast<std::byte *>(genericSoABlocksView.pca().metadata().addressOf_vector_3()));
    REQUIRE(reinterpret_cast<std::byte *>(genericSoABlocksView.pca().metadata().addressOf_vector_3()) +
                cms::soa::alignSize(elemsPCA * sizeof(float), SoAGenericBlocks::alignment) ==
            reinterpret_cast<std::byte *>(genericSoABlocksView.pca().metadata().addressOf_candidateDirection()));

    // Ckeck the correctness of the copy
    for (size_t i = 0; i < elemsPos; i++) {
      REQUIRE(genericSoABlocksView.position()[i].x() == positionConstView[i].x());
      REQUIRE(genericSoABlocksView.position()[i].y() == positionConstView[i].y());
      REQUIRE(genericSoABlocksView.position()[i].z() == positionConstView[i].z());
    }
    REQUIRE(genericSoABlocksView.position().detectorType() == positionConstView.detectorType());
    for (size_t i = 0; i < elemsPCA; i++) {
      REQUIRE(genericSoABlocksView.pca()[i].vector_1() == pcaConstView[i].vector_1());
      REQUIRE(genericSoABlocksView.pca()[i].vector_2() == pcaConstView[i].vector_2());
      REQUIRE(genericSoABlocksView.pca()[i].vector_3() == pcaConstView[i].vector_3());
      REQUIRE(genericSoABlocksView.pca()[i].candidateDirection()(0) == pcaConstView[i].candidateDirection()(0));
      REQUIRE(genericSoABlocksView.pca()[i].candidateDirection()(1) == pcaConstView[i].candidateDirection()(1));
      REQUIRE(genericSoABlocksView.pca()[i].candidateDirection()(2) == pcaConstView[i].candidateDirection()(2));
    }

    // Check for the independency of the aggregated SoA
    genericSoABlocksView.position()[3].x() = 0.;
    REQUIRE(genericSoABlocksView.position()[3].x() != positionView[3].x());
  }
}
