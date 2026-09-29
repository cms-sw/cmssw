#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"

// Similar test to SoAGenericBlocksView but with nested SoABlocks

GENERATE_SOA_LAYOUT(SoALayout1, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))

GENERATE_SOA_LAYOUT(SoALayout2, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))

GENERATE_SOA_LAYOUT(SoALayout3, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))

GENERATE_SOA_BLOCKS(BlocksTemplate, SOA_BLOCK(first, SoALayout1), SOA_BLOCK(second, SoALayout2))

GENERATE_SOA_BLOCKS(NestedBlocksTemplate, SOA_BLOCK(blocks, BlocksTemplate), SOA_BLOCK(soa, SoALayout3))

using BlocksSoA = BlocksTemplate<>;
using BlocksView = BlocksSoA::View;
using BlocksConstView = BlocksSoA::ConstView;

using NestedBlocksSoA = NestedBlocksTemplate<>;
using NestedBlocksView = NestedBlocksSoA::View;
using NestedBlocksConstView = NestedBlocksSoA::ConstView;

TEST_CASE("SoAGenericNestedBlocksView") {
  // different number of elements for the SoAs
  std::array<NestedBlocksSoA::size_type, 3> sizes = {10, 20, 30};

  // buffer sizes
  const auto bufferSize = NestedBlocksSoA::computeDataSize(sizes);

  // memory buffer for the SoA of positions
  std::unique_ptr<std::byte, decltype(std::free) *> buffer{
      reinterpret_cast<std::byte *>(aligned_alloc(NestedBlocksSoA::alignment, bufferSize)), std::free};

  // SoA Layouts
  NestedBlocksSoA nestedBlocks{buffer.get(), sizes};
  NestedBlocksView nestedBlocksView{nestedBlocks};
  NestedBlocksConstView nestedBlocksConstView{nestedBlocks};

  // fill up
  nestedBlocksView.blocks().first().id() = 21;
  for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; i++) {
    nestedBlocksView.blocks().first()[i].column() = static_cast<int>(i);
    nestedBlocksView.blocks().first()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
  }

  nestedBlocksView.blocks().second().id() = 42;
  for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; i++) {
    nestedBlocksView.blocks().second()[i].column() = static_cast<int>(i);
    nestedBlocksView.blocks().second()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
  }

  nestedBlocksView.soa().id() = 666;
  for (NestedBlocksSoA::size_type i = 0; i < sizes[2]; i++) {
    nestedBlocksView.soa()[i].column() = static_cast<int>(i);
    nestedBlocksView.soa()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
  }

  SECTION("GenericBlocks View from nested blocks") {
    // building the SoABlocks View, there is no need for runtime check for the size since they are different
    BlocksView blocksView{nestedBlocksView.blocks().first(), nestedBlocksView.blocks().second()};

    // Verify metadata
    REQUIRE(blocksView.metadata().size()[0] == sizes[0]);
    REQUIRE(blocksView.first().metadata().size() == sizes[0]);
    REQUIRE(blocksView.metadata().size()[1] == sizes[1]);
    REQUIRE(blocksView.second().metadata().size() == sizes[1]);

    // Check for equality of memory addresses
    REQUIRE(blocksView.first().metadata().addressOf_column() ==
            nestedBlocksView.blocks().first().metadata().addressOf_column());
    REQUIRE(blocksView.first().metadata().addressOf_vector() ==
            nestedBlocksView.blocks().first().metadata().addressOf_vector());
    REQUIRE(blocksView.second().metadata().addressOf_column() ==
            nestedBlocksView.blocks().second().metadata().addressOf_column());
    REQUIRE(blocksView.second().metadata().addressOf_vector() ==
            nestedBlocksView.blocks().second().metadata().addressOf_vector());

    // Verify data
    for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
      auto nestedFirst = nestedBlocksView.blocks().first()[i];
      auto first = blocksView.first()[i];
      REQUIRE(first.column() == nestedFirst.column());
      REQUIRE(first.vector() == nestedFirst.vector());
    }

    for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
      auto nestedSecond = nestedBlocksView.blocks().second()[i];
      auto second = blocksView.second()[i];
      REQUIRE(second.column() == nestedSecond.column());
      REQUIRE(second.vector() == nestedSecond.vector());
    }

    REQUIRE(nestedBlocksView.blocks().first().id() == blocksView.first().id());
    REQUIRE(nestedBlocksView.blocks().second().id() == blocksView.second().id());
  }

  SECTION("GenericBlocks ConstView from const nested blocks") {
    // building the SoABlocks ConstView, there is no need for runtime check for the size since they are different
    BlocksConstView blocksConstView{nestedBlocksConstView.blocks().first(), nestedBlocksConstView.blocks().second()};

    // Verify metadata
    REQUIRE(blocksConstView.metadata().size()[0] == sizes[0]);
    REQUIRE(blocksConstView.first().metadata().size() == sizes[0]);
    REQUIRE(blocksConstView.metadata().size()[1] == sizes[1]);
    REQUIRE(blocksConstView.second().metadata().size() == sizes[1]);

    // Check for equality of memory addresses
    REQUIRE(blocksConstView.first().metadata().addressOf_column() ==
            nestedBlocksConstView.blocks().first().metadata().addressOf_column());
    REQUIRE(blocksConstView.first().metadata().addressOf_vector() ==
            nestedBlocksConstView.blocks().first().metadata().addressOf_vector());
    REQUIRE(blocksConstView.second().metadata().addressOf_column() ==
            nestedBlocksConstView.blocks().second().metadata().addressOf_column());
    REQUIRE(blocksConstView.second().metadata().addressOf_vector() ==
            nestedBlocksConstView.blocks().second().metadata().addressOf_vector());

    // Verify data
    for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
      auto nestedFirst = nestedBlocksConstView.blocks().first()[i];
      auto first = blocksConstView.first()[i];
      REQUIRE(first.column() == nestedFirst.column());
      REQUIRE(first.vector() == nestedFirst.vector());
    }

    for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
      auto nestedSecond = nestedBlocksConstView.blocks().second()[i];
      auto second = blocksConstView.second()[i];
      REQUIRE(second.column() == nestedSecond.column());
      REQUIRE(second.vector() == nestedSecond.vector());
    }

    REQUIRE(nestedBlocksConstView.blocks().first().id() == blocksConstView.first().id());
    REQUIRE(nestedBlocksConstView.blocks().second().id() == blocksConstView.second().id());
  }

  SECTION("GenericBlocks ConstView from nested blocks") {
    // building the SoABlocks ConstView, there is no need for runtime check for the size since they are different
    BlocksConstView blocksConstView{nestedBlocksView.blocks().first(), nestedBlocksView.blocks().second()};

    // Verify metadata
    REQUIRE(blocksConstView.metadata().size()[0] == sizes[0]);
    REQUIRE(blocksConstView.first().metadata().size() == sizes[0]);
    REQUIRE(blocksConstView.metadata().size()[1] == sizes[1]);
    REQUIRE(blocksConstView.second().metadata().size() == sizes[1]);

    // Check for equality of memory addresses
    REQUIRE(blocksConstView.first().metadata().addressOf_column() ==
            nestedBlocksConstView.blocks().first().metadata().addressOf_column());
    REQUIRE(blocksConstView.first().metadata().addressOf_vector() ==
            nestedBlocksConstView.blocks().first().metadata().addressOf_vector());
    REQUIRE(blocksConstView.second().metadata().addressOf_column() ==
            nestedBlocksConstView.blocks().second().metadata().addressOf_column());
    REQUIRE(blocksConstView.second().metadata().addressOf_vector() ==
            nestedBlocksConstView.blocks().second().metadata().addressOf_vector());

    // Verify data
    for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
      auto nestedFirst = nestedBlocksConstView.blocks().first()[i];
      auto first = blocksConstView.first()[i];
      REQUIRE(first.column() == nestedFirst.column());
      REQUIRE(first.vector() == nestedFirst.vector());
    }

    for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
      auto nestedSecond = nestedBlocksConstView.blocks().second()[i];
      auto second = blocksConstView.second()[i];
      REQUIRE(second.column() == nestedSecond.column());
      REQUIRE(second.vector() == nestedSecond.vector());
    }

    REQUIRE(nestedBlocksConstView.blocks().first().id() == blocksConstView.first().id());
    REQUIRE(nestedBlocksConstView.blocks().second().id() == blocksConstView.second().id());
  }

  SECTION("Deep copy the nested blocks to a normal blocks layout") {
    // building the SoABlocks View, there is no need for runtime check for the size since they are different
    BlocksView genericBlocksView{nestedBlocksView.blocks().first(), nestedBlocksView.blocks().second()};

    // Instantiate a SoABlocks
    std::array<NestedBlocksSoA::size_type, 2> size = {sizes[0], sizes[1]};
    const std::size_t blocksBufferSize = BlocksSoA::computeDataSize(size);
    std::unique_ptr<std::byte, decltype(std::free) *> bufferBlocks{
        reinterpret_cast<std::byte *>(aligned_alloc(BlocksSoA::alignment, blocksBufferSize)), std::free};

    BlocksSoA genericBlocks{bufferBlocks.get(), size};

    genericBlocks.deepCopy(genericBlocksView);

    BlocksView genericSoABlocksView{genericBlocks};
    // Check for inequality of memory addresses
    REQUIRE(genericSoABlocksView.first().metadata().addressOf_column() !=
            nestedBlocksView.blocks().first().metadata().addressOf_column());
    REQUIRE(genericSoABlocksView.first().metadata().addressOf_vector() !=
            nestedBlocksView.blocks().first().metadata().addressOf_vector());
    REQUIRE(genericSoABlocksView.second().metadata().addressOf_column() !=
            nestedBlocksView.blocks().second().metadata().addressOf_column());
    REQUIRE(genericSoABlocksView.second().metadata().addressOf_vector() !=
            nestedBlocksView.blocks().second().metadata().addressOf_vector());

    // Verify data
    for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
      auto nestedFirst = nestedBlocksConstView.blocks().first()[i];
      auto first = genericSoABlocksView.first()[i];
      REQUIRE(first.column() == nestedFirst.column());
      REQUIRE(first.vector() == nestedFirst.vector());
    }

    for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
      auto nestedSecond = nestedBlocksConstView.blocks().second()[i];
      auto second = genericSoABlocksView.second()[i];
      REQUIRE(second.column() == nestedSecond.column());
      REQUIRE(second.vector() == nestedSecond.vector());
    }

    REQUIRE(nestedBlocksConstView.blocks().first().id() == genericSoABlocksView.first().id());
    REQUIRE(nestedBlocksConstView.blocks().second().id() == genericSoABlocksView.second().id());
  }
}
