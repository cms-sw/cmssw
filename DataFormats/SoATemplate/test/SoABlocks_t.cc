#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"

// This file tests the main properties of SoABlocks

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

GENERATE_SOA_LAYOUT(SoATemplate, SOA_SCALAR(int, id), SOA_SCALAR(int, type), SOA_SCALAR(float, energy))

GENERATE_SOA_BLOCKS(SoABlocksTemplate,
                    SOA_BLOCK(position, SoAPositionTemplate),
                    SOA_BLOCK(pca, SoAPCATemplate),
                    SOA_BLOCK(scalars, SoATemplate))

using SoABlocks = SoABlocksTemplate<>;
// TODO: Implement template version of View and ConstView (ViewTemplate and ConstViewTemplate)
using SoABlocksView = SoABlocks::View;
using SoABlocksConstView = SoABlocks::ConstView;

TEST_CASE("SoABlocks") {
  // Create a SoABlocks instance with three blocks of different sizes
  std::array<cms::soa::size_type, 3> sizes{{10, 20, 1}};
  const std::size_t blocksBufferSize = SoABlocks::computeDataSize(sizes);

  std::unique_ptr<std::byte, decltype(std::free) *> buffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SoABlocks::alignment, blocksBufferSize)), std::free};

  SoABlocks blocks(buffer.get(), sizes);
  SoABlocksView blocksView{blocks};
  SoABlocksConstView blocksConstView{blocks};

  // Verify position data
  REQUIRE(blocks.position().metadata().nextByte() == blocks.metadata().addressOf_pca());
  REQUIRE(blocks.pca().metadata().nextByte() == blocks.metadata().addressOf_scalars());

  // Fill the blocks with some data
  blocksView.position().detectorType() = 1;
  for (int i = 0; i < blocksView.position().metadata().size(); ++i) {
    blocksView.position()[i] = {0.1f, 0.2f, 0.3f};
  }
  for (int i = 0; i < blocksView.metadata().size()[1]; ++i) {
    blocksView.pca()[i].vector_1() = 0.0f;
    blocksView.pca()[i].vector_2() = 0.0f;
    blocksView.pca()[i].vector_3() = 1.0f;
    blocksView.pca()[i].candidateDirection() = Eigen::Vector3d(1.0, 0.0, 0.0);
  }
  blocksView.scalars().id() = 42;
  blocksView.scalars().type() = 1;
  blocksView.scalars().energy() = 100.0f;

  SECTION("SoABlocks View") {
    // Verify metadata
    REQUIRE(blocksView.metadata().size()[0] == 10);
    REQUIRE(blocksView.position().metadata().size() == 10);
    REQUIRE(blocksView.metadata().size()[1] == 20);
    REQUIRE(blocksView.pca().metadata().size() == 20);
    REQUIRE(blocksView.metadata().size()[2] == 1);
    REQUIRE(blocksView.scalars().metadata().size() == 1);

    // Verify data
    for (int i = 0; i < blocksView.position().metadata().size(); ++i) {
      auto pos = blocksView.position()[i];
      REQUIRE(pos.x() == 0.1f);
      REQUIRE(pos.y() == 0.2f);
      REQUIRE(pos.z() == 0.3f);
    }
    for (int i = 0; i < blocksView.pca().metadata().size(); ++i) {
      auto pca = blocksView.pca()[i];
      REQUIRE(pca.vector_1() == 0.0f);
      REQUIRE(pca.vector_2() == 0.0f);
      REQUIRE(pca.vector_3() == 1.0f);
      REQUIRE(pca.candidateDirection()(0) == 1.0);
      REQUIRE(pca.candidateDirection()(1) == 0.0);
      REQUIRE(pca.candidateDirection()(2) == 0.0);
    }
  }

  SECTION("SoABlocks ConstView") {
    // Verify metadata
    REQUIRE(blocksConstView.metadata().size()[0] == 10);
    REQUIRE(blocksConstView.position().metadata().size() == 10);
    REQUIRE(blocksConstView.metadata().size()[1] == 20);
    REQUIRE(blocksConstView.pca().metadata().size() == 20);
    REQUIRE(blocksConstView.metadata().size()[2] == 1);
    REQUIRE(blocksConstView.scalars().metadata().size() == 1);

    // Verify data
    for (int i = 0; i < blocksConstView.position().metadata().size(); ++i) {
      auto pos = blocksConstView.position()[i];
      REQUIRE(pos.x() == 0.1f);
      REQUIRE(pos.y() == 0.2f);
      REQUIRE(pos.z() == 0.3f);
    }
    for (int i = 0; i < blocksConstView.pca().metadata().size(); ++i) {
      auto pca = blocksConstView.pca()[i];
      REQUIRE(pca.vector_1() == 0.0f);
      REQUIRE(pca.vector_2() == 0.0f);
      REQUIRE(pca.vector_3() == 1.0f);
      REQUIRE(pca.candidateDirection()(0) == 1.0);
      REQUIRE(pca.candidateDirection()(1) == 0.0);
      REQUIRE(pca.candidateDirection()(2) == 0.0);
    }
  }

  SECTION("Range checking View") {
    // Range checking is enabled by default
    // TODO: give possibility to disable range checking
    int underflow = -1;
    int overflow = blocksView.position().metadata().size();
    // Check for under-and overflow in the row accessor
    REQUIRE_THROWS_AS(blocksView.position()[underflow], std::out_of_range);
    REQUIRE_THROWS_AS(blocksView.position()[overflow], std::out_of_range);
    // Check for under-and overflow in the element accessors
    REQUIRE_THROWS_AS(blocksView.position().x(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(blocksView.position().x(overflow), std::out_of_range);
  }

  SECTION("Range checking ConstView") {
    // Range checking is enabled by default
    // TODO: give possibility to disable range checking
    int underflow = -1;
    int overflow = blocksConstView.pca().metadata().size();
    // Check for under-and overflow in the row accessor
    REQUIRE_THROWS_AS(blocksConstView.pca()[underflow], std::out_of_range);
    REQUIRE_THROWS_AS(blocksConstView.pca()[overflow], std::out_of_range);
    // Check for under-and overflow in the element accessors
    REQUIRE_THROWS_AS(blocksConstView.pca().vector_1(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(blocksConstView.pca().vector_1(overflow), std::out_of_range);
  }
}
