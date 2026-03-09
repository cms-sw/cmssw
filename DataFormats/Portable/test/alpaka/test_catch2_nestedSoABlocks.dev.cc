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

// TODO remove
#include <iostream>

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace Catch::Matchers;

GENERATE_SOA_LAYOUT(SoALayout1, 
                    SOA_COLUMN(int, column),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, vector),
                    SOA_SCALAR(int, id))

GENERATE_SOA_LAYOUT(SoALayout2, 
                    SOA_COLUMN(int, column),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, vector),
                    SOA_SCALAR(int, id))

GENERATE_SOA_LAYOUT(SoALayout3, 
                    SOA_COLUMN(int, column),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, vector),
                    SOA_SCALAR(int, id))

GENERATE_SOA_LAYOUT(SoALayout4, 
                    SOA_COLUMN(int, column),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, vector),
                    SOA_SCALAR(int, id))

GENERATE_SOA_BLOCKS(BlocksTemplate,
                    SOA_BLOCK(first, SoALayout1),
                    SOA_BLOCK(second, SoALayout2))

GENERATE_SOA_BLOCKS(SingleNestedBlocksTemplate,
                    SOA_BLOCK(blocks, BlocksTemplate),
                    SOA_BLOCK(soa, SoALayout3))

GENERATE_SOA_BLOCKS(DoubleNestedBlocksTemplate,
                    SOA_BLOCK(blocks, SingleNestedBlocksTemplate),
                    SOA_BLOCK(soa, SoALayout4))

using BlockSoA = DoubleNestedBlocksTemplate<>;
using View = BlockSoA::View;
using ConstView = BlockSoA::ConstView;

// Fill SoAs
struct FillSoAs {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, View view) const {
    // Fill elements of SoALayout1
    if (cms::alpakatools::once_per_grid(acc)) {
      view.blocks().blocks().first().id() = view.metadata().size()[0];
    }
    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[0])) {
      auto element = view.blocks().blocks().first()[i];
      element.column() = static_cast<int>(i);
      element.vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    // Fill elements of SoALayout2
    if (cms::alpakatools::once_per_grid(acc)) {
      view.blocks().blocks().second().id() = view.metadata().size()[1];
    }
    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[1])) {
      auto element = view.blocks().blocks().second()[i];
      element.column() = static_cast<int>(i * 10);
      element.vector() = Eigen::Vector3d(i * 10, i * 10 + 1, i * 10 + 2);
    }

    // Fill elements of SoALayout3
    if (cms::alpakatools::once_per_grid(acc)) {
      view.blocks().soa().id() = view.metadata().size()[2];
    }
    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[2])) {
      auto element = view.blocks().soa()[i];
      element.column() = static_cast<int>(i * 100);
      element.vector() = Eigen::Vector3d(i * 100, i * 100 + 1, i * 100 + 2);
    }

    // Fill elements of SoALayout4
    if (cms::alpakatools::once_per_grid(acc)) {
      view.soa().id() = view.metadata().size()[3];
    }
    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[3])) {
      auto element = view.soa()[i];
      element.column() = static_cast<int>(i * 6654);
      element.vector() = Eigen::Vector3d(i * 6654, i * 6654 + 1, i * 6654 + 2);
    }
  }
};

void checkNestedSoABlocks(const ConstView& view) {
    REQUIRE(view.blocks().blocks().first().id() == view.metadata().size()[0]);
    for (BlockSoA::size_type i = 0; i < view.metadata().size()[0]; ++i) {
        const auto& element = view.blocks().blocks().first()[i];
        REQUIRE(element.column() == static_cast<int>(i));
        REQUIRE(element.vector().isApprox(Eigen::Vector3d(i, i + 1, i + 2)));
    }
    
    REQUIRE(view.blocks().blocks().second().id() == view.metadata().size()[1]);
    for (BlockSoA::size_type i = 0; i < view.metadata().size()[1]; ++i) {
        const auto& element = view.blocks().blocks().second()[i];
        REQUIRE(element.column() == static_cast<int>(i * 10));
        REQUIRE(element.vector().isApprox(Eigen::Vector3d(i * 10, i * 10 + 1, i * 10 + 2)));
    }

    REQUIRE(view.blocks().soa().id() == view.metadata().size()[2]);
    for (BlockSoA::size_type i = 0; i < view.metadata().size()[2]; ++i) {
        const auto& element = view.blocks().soa()[i];
        REQUIRE(element.column() == static_cast<int>(i * 100));
        REQUIRE(element.vector().isApprox(Eigen::Vector3d(i * 100, i * 100 + 1, i * 100 + 2)));
    }

    REQUIRE(view.soa().id() == view.metadata().size()[3]);
    for (BlockSoA::size_type i = 0; i < view.metadata().size()[3]; ++i) {
        const auto& element = view.soa()[i];
        REQUIRE(element.column() == static_cast<int>(i * 6654));
        REQUIRE(element.vector().isApprox(Eigen::Vector3d(i * 6654, i * 6654 + 1, i * 6654 + 2)));
    }
}

TEST_CASE("NestedSoABlocks minimal test") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available for the " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << " backend, skipping.\n";
    return;
  }

  for (auto const& device : devices) {

    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    std::array<BlockSoA::size_type, 4> sizes = {2, 1189, 33, 3333};
    const std::size_t N = *std::max_element(sizes.begin(), sizes.end());

    PortableCollection<Device, BlockSoA> nestedBlocksCollection(queue, sizes);
    View view = nestedBlocksCollection.view();

    // check that the sizes in the View are correctly propagated from the SoABlocks
    REQUIRE(view.metadata().size()[0] == sizes[0]);
    REQUIRE(view.blocks().blocks().first().metadata().size() == sizes[0]);
    REQUIRE(view.metadata().size()[1] == sizes[1]);
    REQUIRE(view.blocks().blocks().second().metadata().size() == sizes[1]);
    REQUIRE(view.metadata().size()[2] == sizes[2]);
    REQUIRE(view.blocks().soa().metadata().size() == sizes[2]);
    REQUIRE(view.metadata().size()[3] == sizes[3]);
    REQUIRE(view.soa().metadata().size() == sizes[3]);

    // Work division
    const std::size_t blockSize = 256;
    const std::size_t nBlocks = cms::alpakatools::divide_up_by(N, blockSize);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize);

    // Fill: all layouts
    alpaka::exec<Acc1D>(queue, workDiv, FillSoAs{}, view);
    alpaka::wait(queue);

    // Check results on host
    PortableHostCollection<BlockSoA> nestedBlocksHostCollection(cms::alpakatools::host(), sizes);
    alpaka::memcpy(queue, nestedBlocksHostCollection.buffer(), nestedBlocksCollection.buffer());
    alpaka::wait(queue);

    ConstView constHostView = nestedBlocksHostCollection.const_view();
    checkNestedSoABlocks(constHostView);
  }

}