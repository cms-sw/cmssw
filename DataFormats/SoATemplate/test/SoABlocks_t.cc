#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"

using namespace Catch::Matchers;

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

GENERATE_SOA_LAYOUT(
    SimpleLayoutTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z), SOA_COLUMN(float, t))

GENERATE_SOA_BLOCKS(SoABlocksTemplate,
                    SOA_BLOCK(position, SoAPositionTemplate),
                    SOA_BLOCK(pca, SoAPCATemplate),
                    SOA_BLOCK(scalars, SoATemplate))

GENERATE_SOA_BLOCKS(NestedBlocksTemplate, SOA_BLOCK(blocks, SoABlocksTemplate), SOA_BLOCK(simple, SimpleLayoutTemplate))

using SoABlocks = SoABlocksTemplate<>;
using SoABlocksView = SoABlocks::View;
using SoABlocksConstView = SoABlocks::ConstView;

using NestedBlocks = NestedBlocksTemplate<>;
using NestedBlocksView = NestedBlocks::View;
using NestedBlocksConstView = NestedBlocks::ConstView;

TEST_CASE("SoABlocks") {
  // Create a SoABlocks instance with three blocks of different sizes
  std::array<cms::soa::size_type, 3> sizes{{10, 20, 1}};
  const std::size_t blocksBufferSize = SoABlocks::computeDataSize(sizes);

  std::unique_ptr<std::byte, decltype(std::free) *> buffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SoABlocks::alignment, blocksBufferSize)), std::free};

  SoABlocks blocks(buffer.get(), sizes);
  SoABlocksView blocksView{blocks};
  SoABlocksConstView blocksConstView{blocks};

  REQUIRE(SoABlocks::alignment == cms::soa::CacheLineSize::defaultSize);
  REQUIRE(SoABlocks::alignmentEnforcement == cms::soa::AlignmentEnforcement::relaxed);

  REQUIRE(blocks.position().alignment == cms::soa::CacheLineSize::defaultSize);
  REQUIRE(blocks.position().alignmentEnforcement == cms::soa::AlignmentEnforcement::relaxed);

  REQUIRE(blocks.pca().alignment == cms::soa::CacheLineSize::defaultSize);
  REQUIRE(blocks.pca().alignmentEnforcement == cms::soa::AlignmentEnforcement::relaxed);

  REQUIRE(blocks.scalars().alignment == cms::soa::CacheLineSize::defaultSize);
  REQUIRE(blocks.scalars().alignmentEnforcement == cms::soa::AlignmentEnforcement::relaxed);

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

  SECTION("Check template parameters") {
    static constexpr cms::soa::byte_size_type testAlignment = 256;
    static constexpr bool alignmentEnforcement = cms::soa::AlignmentEnforcement::enforced;

    using SoABlocksTemplated = SoABlocksTemplate<testAlignment, alignmentEnforcement>;

    std::array<cms::soa::size_type, 3> sizes{{10, 20, 1}};
    const std::size_t blocksBufferSize = SoABlocksTemplated::computeDataSize(sizes);

    std::unique_ptr<std::byte, decltype(std::free) *> buffer{
        reinterpret_cast<std::byte *>(aligned_alloc(SoABlocksTemplated::alignment, blocksBufferSize)), std::free};

    SoABlocksTemplated blocksTemplated(buffer.get(), sizes);

    REQUIRE(SoABlocksTemplated::alignment == testAlignment);
    REQUIRE(SoABlocksTemplated::alignmentEnforcement == alignmentEnforcement);

    REQUIRE(blocksTemplated.position().alignment == testAlignment);
    REQUIRE(blocksTemplated.position().alignmentEnforcement == alignmentEnforcement);

    REQUIRE(blocksTemplated.pca().alignment == testAlignment);
    REQUIRE(blocksTemplated.pca().alignmentEnforcement == alignmentEnforcement);

    REQUIRE(blocksTemplated.scalars().alignment == testAlignment);
    REQUIRE(blocksTemplated.scalars().alignmentEnforcement == alignmentEnforcement);
  }

  SECTION("Check view template parameters") {
    using NoRangeCheckBlockView =
        SoABlocks::ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::disabled>;
    NoRangeCheckBlockView noRangeCheckBlockView{blocks};

    REQUIRE(noRangeCheckBlockView.restrictQualify == cms::soa::RestrictQualify::Default);
    REQUIRE(noRangeCheckBlockView.rangeChecking == cms::soa::RangeChecking::disabled);
    REQUIRE(noRangeCheckBlockView.position().restrictQualify == cms::soa::RestrictQualify::Default);
    REQUIRE(noRangeCheckBlockView.position().rangeChecking == cms::soa::RangeChecking::disabled);
    REQUIRE(noRangeCheckBlockView.pca().restrictQualify == cms::soa::RestrictQualify::Default);
    REQUIRE(noRangeCheckBlockView.pca().rangeChecking == cms::soa::RangeChecking::disabled);
    REQUIRE(noRangeCheckBlockView.scalars().restrictQualify == cms::soa::RestrictQualify::Default);
    REQUIRE(noRangeCheckBlockView.scalars().rangeChecking == cms::soa::RangeChecking::disabled);

    using NoRestrictBlockView =
        SoABlocks::ViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::Default>;
    NoRestrictBlockView noRestrictBlockView{blocks};

    REQUIRE(noRestrictBlockView.restrictQualify == cms::soa::RestrictQualify::disabled);
    REQUIRE(noRestrictBlockView.rangeChecking == cms::soa::RangeChecking::Default);
    REQUIRE(noRestrictBlockView.position().restrictQualify == cms::soa::RestrictQualify::disabled);
    REQUIRE(noRestrictBlockView.position().rangeChecking == cms::soa::RangeChecking::Default);
    REQUIRE(noRestrictBlockView.pca().restrictQualify == cms::soa::RestrictQualify::disabled);
    REQUIRE(noRestrictBlockView.pca().rangeChecking == cms::soa::RangeChecking::Default);
    REQUIRE(noRestrictBlockView.scalars().restrictQualify == cms::soa::RestrictQualify::disabled);
    REQUIRE(noRestrictBlockView.scalars().rangeChecking == cms::soa::RangeChecking::Default);

    using NoRangeCheckBlockConstView =
        SoABlocks::ConstViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::disabled>;
    NoRangeCheckBlockConstView noRangeCheckBlockConstView{blocks};

    REQUIRE(noRangeCheckBlockConstView.restrictQualify == cms::soa::RestrictQualify::Default);
    REQUIRE(noRangeCheckBlockConstView.rangeChecking == cms::soa::RangeChecking::disabled);
    REQUIRE(noRangeCheckBlockConstView.position().restrictQualify == cms::soa::RestrictQualify::Default);
    REQUIRE(noRangeCheckBlockConstView.position().rangeChecking == cms::soa::RangeChecking::disabled);
    REQUIRE(noRangeCheckBlockConstView.pca().restrictQualify == cms::soa::RestrictQualify::Default);
    REQUIRE(noRangeCheckBlockConstView.pca().rangeChecking == cms::soa::RangeChecking::disabled);
    REQUIRE(noRangeCheckBlockConstView.scalars().restrictQualify == cms::soa::RestrictQualify::Default);
    REQUIRE(noRangeCheckBlockConstView.scalars().rangeChecking == cms::soa::RangeChecking::disabled);

    using NoRestrictBlockConstView =
        SoABlocks::ConstViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::Default>;
    NoRestrictBlockConstView noRestrictBlockConstView{blocks};

    REQUIRE(noRestrictBlockConstView.restrictQualify == cms::soa::RestrictQualify::disabled);
    REQUIRE(noRestrictBlockConstView.rangeChecking == cms::soa::RangeChecking::Default);
    REQUIRE(noRestrictBlockConstView.position().restrictQualify == cms::soa::RestrictQualify::disabled);
    REQUIRE(noRestrictBlockConstView.position().rangeChecking == cms::soa::RangeChecking::Default);
    REQUIRE(noRestrictBlockConstView.pca().restrictQualify == cms::soa::RestrictQualify::disabled);
    REQUIRE(noRestrictBlockConstView.pca().rangeChecking == cms::soa::RangeChecking::Default);
    REQUIRE(noRestrictBlockConstView.scalars().restrictQualify == cms::soa::RestrictQualify::disabled);
    REQUIRE(noRestrictBlockConstView.scalars().rangeChecking == cms::soa::RangeChecking::Default);
  }

  SECTION("Check extended blocks layout") {
    std::array<cms::soa::size_type, 4> sizes{{11, 12, 13, 14}};
    const std::size_t blocksExtendedBufferSize = NestedBlocks::computeDataSize(sizes);

    std::unique_ptr<std::byte, decltype(std::free) *> buffer{
        reinterpret_cast<std::byte *>(aligned_alloc(NestedBlocks::alignment, blocksExtendedBufferSize)), std::free};

    NestedBlocks NestedBlocksSoA(buffer.get(), sizes);
    NestedBlocksView nestedBlocksView{NestedBlocksSoA};
    NestedBlocksConstView nestedBlocksConstView{NestedBlocksSoA};

    nestedBlocksView.blocks().position().detectorType() = 1;
    for (int i = 0; i < nestedBlocksView.metadata().size()[0]; ++i) {
      nestedBlocksView.blocks().position()[i] = {0.1f, 0.2f, 0.3f};
    }

    for (int i = 0; i < nestedBlocksView.metadata().size()[1]; ++i) {
      nestedBlocksView.blocks().pca()[i].vector_1() = 0.0f;
      nestedBlocksView.blocks().pca()[i].vector_2() = 0.0f;
      nestedBlocksView.blocks().pca()[i].vector_3() = 1.0f;
      nestedBlocksView.blocks().pca()[i].candidateDirection() = Eigen::Vector3d(1.0, 0.0, 0.0);
    }
    nestedBlocksView.blocks().scalars().id() = 42;
    nestedBlocksView.blocks().scalars().type() = 1;
    nestedBlocksView.blocks().scalars().energy() = 100.0f;

    for (int i = 0; i < nestedBlocksView.metadata().size()[3]; ++i) {
      nestedBlocksView.simple()[i] = {2.1f, 2.2f, 2.3f, 2.4f};
    }

    REQUIRE(NestedBlocksSoA.blocks().position().metadata().size() == 11);
    REQUIRE(NestedBlocksSoA.blocks().pca().metadata().size() == 12);
    REQUIRE(NestedBlocksSoA.blocks().scalars().metadata().size() == 13);
    REQUIRE(NestedBlocksSoA.simple().metadata().size() == 14);

    REQUIRE(nestedBlocksConstView.blocks().position().detectorType() == 1);
    for (int i = 0; i < nestedBlocksConstView.metadata().size()[0]; ++i) {
      REQUIRE_THAT(nestedBlocksConstView.blocks().position()[i].x(), WithinRel(0.1f));
      REQUIRE_THAT(nestedBlocksConstView.blocks().position()[i].y(), WithinRel(0.2f));
      REQUIRE_THAT(nestedBlocksConstView.blocks().position()[i].z(), WithinRel(0.3f));
    }

    for (int i = 0; i < nestedBlocksConstView.metadata().size()[1]; ++i) {
      REQUIRE_THAT(nestedBlocksConstView.blocks().pca()[i].vector_1(), WithinRel(0.0f));
      REQUIRE_THAT(nestedBlocksConstView.blocks().pca()[i].vector_2(), WithinRel(0.0f));
      REQUIRE_THAT(nestedBlocksConstView.blocks().pca()[i].vector_3(), WithinRel(1.0f));
      REQUIRE_THAT(nestedBlocksConstView.blocks().pca()[i].candidateDirection()[0], WithinRel(1.0));
      REQUIRE_THAT(nestedBlocksConstView.blocks().pca()[i].candidateDirection()[1], WithinRel(0.0));
      REQUIRE_THAT(nestedBlocksConstView.blocks().pca()[i].candidateDirection()[2], WithinRel(0.0));
    }
    REQUIRE(nestedBlocksConstView.blocks().scalars().id() == 42);
    REQUIRE(nestedBlocksConstView.blocks().scalars().type() == 1);
    REQUIRE_THAT(nestedBlocksConstView.blocks().scalars().energy(), WithinRel(100.0f));

    for (int i = 0; i < nestedBlocksConstView.metadata().size()[3]; ++i) {
      REQUIRE_THAT(nestedBlocksConstView.simple()[i].x(), WithinRel(2.1f));
      REQUIRE_THAT(nestedBlocksConstView.simple()[i].y(), WithinRel(2.2f));
      REQUIRE_THAT(nestedBlocksConstView.simple()[i].z(), WithinRel(2.3f));
      REQUIRE_THAT(nestedBlocksConstView.simple()[i].t(), WithinRel(2.4f));
    }
  }
}
