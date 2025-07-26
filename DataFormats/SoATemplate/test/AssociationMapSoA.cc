#include <Eigen/Core>
#include <Eigen/Dense>
#include <span>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

// This file tests the methods of SoABlocks View and ConstView
// A very simple association map using SoABlocks is defined here, with two blocks:
// - indexes: a block containing the indexes of the elements
// - offsets: a block containing the offsets of the elements
// The methods are operators [] that return a std::span of the indexes for a given offset.
// N.B. SOA_VIEW_METHODS and SOA_CONST_VIEW_METHODS macros are available only for SoABlocks.

GENERATE_SOA_LAYOUT(SoAIndexesTemplate, SOA_COLUMN(std::size_t, indexes))

GENERATE_SOA_LAYOUT(SoAOffsetsTemplate, SOA_COLUMN(std::size_t, offsets))

GENERATE_SOA_BLOCKS(SoABlocksTemplate,
                    SOA_BLOCK(indexes, SoAIndexesTemplate),
                    SOA_BLOCK(offsets, SoAOffsetsTemplate),

                    SOA_VIEW_METHODS(std::span<std::size_t> operator[](std::size_t i) {
                      std::size_t start = offsets()[i].offsets();
                      std::size_t end = (i + 1 < static_cast<std::size_t>(offsets().metadata().size()))
                                            ? offsets()[i + 1].offsets()
                                            : indexes().metadata().size();
                      return {indexes().indexes() + start, indexes().indexes() + end};
                    }),

                    SOA_CONST_VIEW_METHODS(const std::span<std::size_t> operator[](std::size_t i) const {
                      std::size_t start = offsets()[i].offsets();
                      std::size_t end = (i + 1 < offsets().metadata().size()) ? offsets()[i + 1].offsets()
                                                                              : indexes().metadata().size();
                      return {indexes().indexes() + start, indexes().indexes() + end};
                    }))

using SoABlocks = SoABlocksTemplate<>;
using SoABlocksView = SoABlocks::View;
using SoABlocksConstView = SoABlocks::ConstView;

TEST_CASE("SoABlocks methods for the (Const)View") {
  // Create a SoABlocks instance with three blocks of different sizes
  std::array<cms::soa::size_type, 2> sizes{{20, 3}};

  const std::size_t blocksBufferSize = SoABlocks::computeDataSize(sizes);

  std::unique_ptr<std::byte, decltype(std::free) *> buffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SoABlocks::alignment, blocksBufferSize)), std::free};

  SoABlocks blocks(buffer.get(), sizes);
  SoABlocksView blocksView{blocks};
  SoABlocksConstView blocksConstView{blocks};

  // Fill the blocks with some data
  for (int i = 0; i < blocksView.indexes().metadata().size(); ++i) {
    blocksView.indexes()[i].indexes() = i;
  }

  blocksView.offsets()[0].offsets() = 0;
  blocksView.offsets()[1].offsets() = 5;
  blocksView.offsets()[2].offsets() = 10;

  SECTION("std::span map with View") {
    // Access the ranges of indexes using the operator[]
    std::span<std::size_t> values_first = blocksView[0];
    std::span<std::size_t> values_second = blocksView[1];
    std::span<std::size_t> values_third = blocksView[2];

    // Verify the values
    for (std::size_t i = 0; i < values_first.size(); ++i) {
      REQUIRE(values_first[i] == i);
    }

    for (std::size_t i = 0; i < values_second.size(); ++i) {
      REQUIRE(values_second[i] == i + 5);
    }

    for (std::size_t i = 0; i < values_third.size(); ++i) {
      REQUIRE(values_third[i] == i + 10);
    }

    // Swap the contents of the first and second spans
    for (std::size_t i = 0; i < values_first.size(); ++i) {
      std::swap(values_first[i], values_second[i]);
    }

    // Verify the values
    for (std::size_t i = 0; i < values_first.size(); ++i) {
      REQUIRE(values_first[i] == i + 5);
    }

    for (std::size_t i = 0; i < values_second.size(); ++i) {
      REQUIRE(values_second[i] == i);
    }
  }

  SECTION("std::span map with ConstView") {
    // Access the ranges of indexes using the operator[]
    const std::span<std::size_t> values_first = blocksView[0];
    const std::span<std::size_t> values_second = blocksView[1];
    const std::span<std::size_t> values_third = blocksView[2];

    // Verify the values
    for (std::size_t i = 0; i < values_first.size(); ++i) {
      REQUIRE(values_first[i] == i);
    }

    for (std::size_t i = 0; i < values_second.size(); ++i) {
      REQUIRE(values_second[i] == i + 5);
    }

    for (std::size_t i = 0; i < values_third.size(); ++i) {
      REQUIRE(values_third[i] == i + 10);
    }
  }
}
