#include <Eigen/Core>
#include <Eigen/Dense>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include <algorithm>

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

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

// Fill SoAs
struct FillSoA {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, NestedBlocksView view) const {
    if (cms::alpakatools::once_per_grid(acc)) {
      view.blocks().first().id() = 21;
      view.blocks().second().id() = 42;
      view.soa().id() = 666;
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[0])) {
      view.blocks().first()[i].column() = static_cast<int>(i);
      view.blocks().first()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[1])) {
      view.blocks().second()[i].column() = static_cast<int>(i);
      view.blocks().second()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[2])) {
      view.soa()[i].column() = static_cast<int>(i);
      view.soa()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }
  }
};

void check(NestedBlocksConstView nestedBlocksConstView, BlocksConstView genericSoABlocksView) {
  REQUIRE(nestedBlocksConstView.metadata().size()[0] == genericSoABlocksView.metadata().size()[0]);
  REQUIRE(nestedBlocksConstView.metadata().size()[1] == genericSoABlocksView.metadata().size()[1]);
  // Verify data
  for (NestedBlocksSoA::size_type i = 0; i < genericSoABlocksView.metadata().size()[0]; ++i) {
    auto nestedFirst = nestedBlocksConstView.blocks().first()[i];
    auto first = genericSoABlocksView.first()[i];
    REQUIRE(first.column() == nestedFirst.column());
    REQUIRE(first.vector() == nestedFirst.vector());
  }

  for (NestedBlocksSoA::size_type i = 0; i < genericSoABlocksView.metadata().size()[0]; ++i) {
    auto nestedSecond = nestedBlocksConstView.blocks().second()[i];
    auto second = genericSoABlocksView.second()[i];
    REQUIRE(second.column() == nestedSecond.column());
    REQUIRE(second.vector() == nestedSecond.vector());
  }

  REQUIRE(nestedBlocksConstView.blocks().first().id() == genericSoABlocksView.first().id());
  REQUIRE(nestedBlocksConstView.blocks().second().id() == genericSoABlocksView.second().id());
}

TEST_CASE("Heterogeneous Deep Copy SoABlocks") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available for the " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << " backend, skipping.\n";
    return;
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    std::array<NestedBlocksSoA::size_type, 3> sizes = {21, 45, 137};

    PortableCollection<Device, NestedBlocksSoA> nestedBlocksCollection(queue, sizes);
    NestedBlocksView nestedBlocksView = nestedBlocksCollection.view();
    NestedBlocksConstView nestedBlocksConstView = nestedBlocksCollection.const_view();

    // fill up
    auto blockSize = 64;
    NestedBlocksSoA::size_type largestSize = *std::max_element(sizes.begin(), sizes.end());
    auto numberOfBlocks = cms::alpakatools::divide_up_by(largestSize, blockSize);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDiv, FillSoA{}, nestedBlocksView);
    alpaka::wait(queue);

    PortableHostCollection<NestedBlocksSoA> h_nestedBlocksCollection(queue, sizes);
    alpaka::memcpy(queue, h_nestedBlocksCollection.buffer(), nestedBlocksCollection.buffer());

    SECTION("Deep copy the View host to host and device to device") {
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

      // PortableCollection that will host the aggregated columns
      PortableCollection<Device, BlocksSoA> blocksCollection(queue, sizes[0], sizes[1]);
      blocksCollection.deepCopy(queue, blocksView);

      BlocksView copiedBlocksView = blocksCollection.view();
      REQUIRE(copiedBlocksView.first().metadata().addressOf_column() !=
              nestedBlocksView.blocks().first().metadata().addressOf_column());
      REQUIRE(copiedBlocksView.first().metadata().addressOf_vector() !=
              nestedBlocksView.blocks().first().metadata().addressOf_vector());
      REQUIRE(copiedBlocksView.second().metadata().addressOf_column() !=
              nestedBlocksView.blocks().second().metadata().addressOf_column());
      REQUIRE(copiedBlocksView.second().metadata().addressOf_vector() !=
              nestedBlocksView.blocks().second().metadata().addressOf_vector());

      PortableHostCollection<BlocksSoA> outputHost(cms::alpakatools::host(), sizes[0], sizes[1]);
      alpaka::memcpy(queue, outputHost.buffer(), blocksCollection.buffer());
      alpaka::wait(queue);

      check(h_nestedBlocksCollection.const_view(), outputHost.const_view());
    }

    SECTION("Deep copy the ConstView host to host and device to device") {
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

      // PortableCollection that will host the aggregated columns
      PortableCollection<Device, BlocksSoA> blocksCollection(queue, sizes[0], sizes[1]);
      blocksCollection.deepCopy(queue, blocksConstView);

      BlocksConstView copiedBlocksConstView = blocksCollection.const_view();
      REQUIRE(copiedBlocksConstView.first().metadata().addressOf_column() !=
              nestedBlocksConstView.blocks().first().metadata().addressOf_column());
      REQUIRE(copiedBlocksConstView.first().metadata().addressOf_vector() !=
              nestedBlocksConstView.blocks().first().metadata().addressOf_vector());
      REQUIRE(copiedBlocksConstView.second().metadata().addressOf_column() !=
              nestedBlocksConstView.blocks().second().metadata().addressOf_column());
      REQUIRE(copiedBlocksConstView.second().metadata().addressOf_vector() !=
              nestedBlocksConstView.blocks().second().metadata().addressOf_vector());

      PortableHostCollection<BlocksSoA> outputHost(cms::alpakatools::host(), sizes[0], sizes[1]);
      alpaka::memcpy(queue, outputHost.buffer(), blocksCollection.buffer());
      alpaka::wait(queue);

      check(h_nestedBlocksCollection.const_view(), outputHost.const_view());
    }

    SECTION("Deep copy the ConstView device to host") {
      BlocksConstView blocksConstView{nestedBlocksConstView.blocks().first(), nestedBlocksConstView.blocks().second()};

      // PortableCollection that will host the aggregated columns
      PortableHostCollection<BlocksSoA> blocksCollection(queue, sizes[0], sizes[1]);
      blocksCollection.deepCopy(queue, blocksConstView);
      alpaka::wait(queue);

      BlocksConstView copiedBlocksConstView = blocksCollection.const_view();
      REQUIRE(copiedBlocksConstView.first().metadata().addressOf_column() !=
              nestedBlocksConstView.blocks().first().metadata().addressOf_column());
      REQUIRE(copiedBlocksConstView.first().metadata().addressOf_vector() !=
              nestedBlocksConstView.blocks().first().metadata().addressOf_vector());
      REQUIRE(copiedBlocksConstView.second().metadata().addressOf_column() !=
              nestedBlocksConstView.blocks().second().metadata().addressOf_column());
      REQUIRE(copiedBlocksConstView.second().metadata().addressOf_vector() !=
              nestedBlocksConstView.blocks().second().metadata().addressOf_vector());

      check(h_nestedBlocksCollection.const_view(), blocksCollection.const_view());
    }

    SECTION("Deep copy the ConstView host to device") {
      BlocksConstView blocksConstView{h_nestedBlocksCollection.const_view().blocks().first(),
                                      h_nestedBlocksCollection.const_view().blocks().second()};

      // PortableCollection that will host the aggregated columns
      PortableCollection<Device, BlocksSoA> blocksCollection(queue, sizes[0], sizes[1]);
      blocksCollection.deepCopy(queue, blocksConstView);
      alpaka::wait(queue);

      BlocksConstView copiedBlocksConstView = blocksCollection.const_view();
      REQUIRE(copiedBlocksConstView.first().metadata().addressOf_column() !=
              h_nestedBlocksCollection.const_view().blocks().first().metadata().addressOf_column());
      REQUIRE(copiedBlocksConstView.first().metadata().addressOf_vector() !=
              h_nestedBlocksCollection.const_view().blocks().first().metadata().addressOf_vector());
      REQUIRE(copiedBlocksConstView.second().metadata().addressOf_column() !=
              h_nestedBlocksCollection.const_view().blocks().second().metadata().addressOf_column());
      REQUIRE(copiedBlocksConstView.second().metadata().addressOf_vector() !=
              h_nestedBlocksCollection.const_view().blocks().second().metadata().addressOf_vector());

      PortableHostCollection<BlocksSoA> outputHost(cms::alpakatools::host(), sizes[0], sizes[1]);
      alpaka::memcpy(queue, outputHost.buffer(), blocksCollection.buffer());
      alpaka::wait(queue);

      check(h_nestedBlocksCollection.const_view(), outputHost.const_view());
    }
  }
}
