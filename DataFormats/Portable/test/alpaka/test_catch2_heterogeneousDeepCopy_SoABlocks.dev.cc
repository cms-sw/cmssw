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
GENERATE_SOA_LAYOUT(SoALayout4, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))

GENERATE_SOA_BLOCKS(FirstBlocksTemplate, SOA_BLOCK(first, SoALayout1), SOA_BLOCK(second, SoALayout2))
GENERATE_SOA_BLOCKS(SecondBlocksTemplate, SOA_BLOCK(first, SoALayout3), SOA_BLOCK(second, SoALayout4))

GENERATE_SOA_BLOCKS(NestedBlocksTemplate,
                    SOA_BLOCK(firstBlocks, FirstBlocksTemplate),
                    SOA_BLOCK(secondBlocks, SecondBlocksTemplate),
                    SOA_BLOCK(firstLayout, SoALayout1),
                    SOA_BLOCK(secondLayout, SoALayout4))

GENERATE_SOA_BLOCKS(GenericBlocksTemplate, SOA_BLOCK(blocks, FirstBlocksTemplate), SOA_BLOCK(layout, SoALayout4))

using NestedBlocksSoA = NestedBlocksTemplate<>;
using NestedBlocksView = NestedBlocksSoA::View;
using NestedBlocksConstView = NestedBlocksSoA::ConstView;

using GenericSoA = GenericBlocksTemplate<>;
using GenericSoAView = GenericSoA::View;
using GenericSoAConstView = GenericSoA::ConstView;

// Fill SoAs
struct FillSoA {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, NestedBlocksView view) const {
    if (cms::alpakatools::once_per_grid(acc)) {
      view.firstBlocks().first().id() = 21;
      view.firstBlocks().second().id() = 22;
      view.secondBlocks().first().id() = 42;
      view.secondBlocks().second().id() = 43;
      view.firstLayout().id() = 333;
      view.secondLayout().id() = 666;
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[0])) {
      view.firstBlocks().first()[i].column() = static_cast<int>(i);
      view.firstBlocks().first()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[1])) {
      view.firstBlocks().second()[i].column() = static_cast<int>(i);
      view.firstBlocks().second()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[2])) {
      view.secondBlocks().first()[i].column() = static_cast<int>(i);
      view.secondBlocks().first()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[3])) {
      view.secondBlocks().second()[i].column() = static_cast<int>(i);
      view.secondBlocks().second()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[4])) {
      view.firstLayout()[i].column() = static_cast<int>(i);
      view.firstLayout()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size()[5])) {
      view.secondLayout()[i].column() = static_cast<int>(i);
      view.secondLayout()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }
  }
};

void check(NestedBlocksConstView nestedBlocksConstView, GenericSoAConstView genericSoABlocksView) {
  REQUIRE(nestedBlocksConstView.metadata().size()[0] == genericSoABlocksView.metadata().size()[0]);
  REQUIRE(nestedBlocksConstView.metadata().size()[1] == genericSoABlocksView.metadata().size()[1]);
  REQUIRE(nestedBlocksConstView.metadata().size()[5] == genericSoABlocksView.metadata().size()[2]);
  // Verify data
  for (NestedBlocksSoA::size_type i = 0; i < genericSoABlocksView.metadata().size()[0]; ++i) {
    auto nestedFirst = nestedBlocksConstView.firstBlocks().first()[i];
    auto first = genericSoABlocksView.blocks().first()[i];
    REQUIRE(first.column() == nestedFirst.column());
    REQUIRE(first.vector() == nestedFirst.vector());
  }

  for (NestedBlocksSoA::size_type i = 0; i < genericSoABlocksView.metadata().size()[1]; ++i) {
    auto nestedSecond = nestedBlocksConstView.firstBlocks().second()[i];
    auto second = genericSoABlocksView.blocks().second()[i];
    REQUIRE(second.column() == nestedSecond.column());
    REQUIRE(second.vector() == nestedSecond.vector());
  }

  for (NestedBlocksSoA::size_type i = 0; i < genericSoABlocksView.metadata().size()[2]; ++i) {
    auto nested = nestedBlocksConstView.secondLayout()[i];
    auto generic = genericSoABlocksView.layout()[i];
    REQUIRE(generic.column() == nested.column());
    REQUIRE(generic.vector() == nested.vector());
  }

  REQUIRE(nestedBlocksConstView.firstBlocks().first().id() == genericSoABlocksView.blocks().first().id());
  REQUIRE(nestedBlocksConstView.firstBlocks().second().id() == genericSoABlocksView.blocks().second().id());
  REQUIRE(nestedBlocksConstView.secondLayout().id() == genericSoABlocksView.layout().id());
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

    std::array<NestedBlocksSoA::size_type, 6> sizes = {21, 45, 137, 43, 222, 177};

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
      GenericSoAView genericView{nestedBlocksView.firstBlocks(), nestedBlocksView.secondLayout()};

      // Verify metadata
      REQUIRE(genericView.metadata().size()[0] == sizes[0]);
      REQUIRE(genericView.blocks().first().metadata().size() == sizes[0]);
      REQUIRE(genericView.metadata().size()[1] == sizes[1]);
      REQUIRE(genericView.blocks().second().metadata().size() == sizes[1]);
      REQUIRE(genericView.metadata().size()[2] == sizes[5]);
      REQUIRE(genericView.layout().metadata().size() == sizes[5]);

      // Check for equality of memory addresses
      REQUIRE(genericView.blocks().first().metadata().addressOf_column() ==
              nestedBlocksView.firstBlocks().first().metadata().addressOf_column());
      REQUIRE(genericView.blocks().first().metadata().addressOf_vector() ==
              nestedBlocksView.firstBlocks().first().metadata().addressOf_vector());
      REQUIRE(genericView.blocks().second().metadata().addressOf_column() ==
              nestedBlocksView.firstBlocks().second().metadata().addressOf_column());
      REQUIRE(genericView.blocks().second().metadata().addressOf_vector() ==
              nestedBlocksView.firstBlocks().second().metadata().addressOf_vector());
      REQUIRE(genericView.layout().metadata().addressOf_column() ==
              nestedBlocksView.secondLayout().metadata().addressOf_column());
      REQUIRE(genericView.layout().metadata().addressOf_vector() ==
              nestedBlocksView.secondLayout().metadata().addressOf_vector());

      // PortableCollection that will host the aggregated columns
      PortableCollection<Device, GenericSoA> blocksCollection(queue, sizes[0], sizes[1], sizes[5]);
      blocksCollection.deepCopy(queue, genericView);

      GenericSoAView copiedBlocksView = blocksCollection.view();
      REQUIRE(copiedBlocksView.blocks().first().metadata().addressOf_column() !=
              nestedBlocksView.firstBlocks().first().metadata().addressOf_column());
      REQUIRE(copiedBlocksView.blocks().first().metadata().addressOf_vector() !=
              nestedBlocksView.firstBlocks().first().metadata().addressOf_vector());
      REQUIRE(copiedBlocksView.blocks().second().metadata().addressOf_column() !=
              nestedBlocksView.firstBlocks().second().metadata().addressOf_column());
      REQUIRE(copiedBlocksView.blocks().second().metadata().addressOf_vector() !=
              nestedBlocksView.firstBlocks().second().metadata().addressOf_vector());
      REQUIRE(copiedBlocksView.layout().metadata().addressOf_column() !=
              nestedBlocksView.secondLayout().metadata().addressOf_column());
      REQUIRE(copiedBlocksView.layout().metadata().addressOf_vector() !=
              nestedBlocksView.secondLayout().metadata().addressOf_vector());

      PortableHostCollection<GenericSoA> outputHost(cms::alpakatools::host(), sizes[0], sizes[1], sizes[5]);
      alpaka::memcpy(queue, outputHost.buffer(), blocksCollection.buffer());
      alpaka::wait(queue);

      check(h_nestedBlocksCollection.const_view(), outputHost.const_view());
    }

    SECTION("Deep copy the ConstView host to host and device to device") {
      GenericSoAConstView genericConstView{nestedBlocksConstView.firstBlocks(), nestedBlocksConstView.secondLayout()};

      // Verify metadata
      REQUIRE(genericConstView.metadata().size()[0] == sizes[0]);
      REQUIRE(genericConstView.blocks().first().metadata().size() == sizes[0]);
      REQUIRE(genericConstView.metadata().size()[1] == sizes[1]);
      REQUIRE(genericConstView.blocks().second().metadata().size() == sizes[1]);
      REQUIRE(genericConstView.metadata().size()[2] == sizes[5]);
      REQUIRE(genericConstView.layout().metadata().size() == sizes[5]);

      // Check for equality of memory addresses
      REQUIRE(genericConstView.blocks().first().metadata().addressOf_column() ==
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_column());
      REQUIRE(genericConstView.blocks().first().metadata().addressOf_vector() ==
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_vector());
      REQUIRE(genericConstView.blocks().second().metadata().addressOf_column() ==
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_column());
      REQUIRE(genericConstView.blocks().second().metadata().addressOf_vector() ==
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_vector());
      REQUIRE(genericConstView.layout().metadata().addressOf_column() ==
              nestedBlocksConstView.secondLayout().metadata().addressOf_column());
      REQUIRE(genericConstView.layout().metadata().addressOf_vector() ==
              nestedBlocksConstView.secondLayout().metadata().addressOf_vector());

      // PortableCollection that will host the aggregated columns
      PortableCollection<Device, GenericSoA> genericCollection(queue, sizes[0], sizes[1], sizes[5]);
      genericCollection.deepCopy(queue, genericConstView);

      GenericSoAConstView copiedGenericConstView = genericCollection.const_view();
      REQUIRE(copiedGenericConstView.blocks().first().metadata().addressOf_column() !=
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.blocks().first().metadata().addressOf_vector() !=
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_vector());
      REQUIRE(copiedGenericConstView.blocks().second().metadata().addressOf_column() !=
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.blocks().second().metadata().addressOf_vector() !=
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_vector());
      REQUIRE(copiedGenericConstView.layout().metadata().addressOf_column() !=
              nestedBlocksConstView.secondLayout().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.layout().metadata().addressOf_vector() !=
              nestedBlocksConstView.secondLayout().metadata().addressOf_vector());

      PortableHostCollection<GenericSoA> outputHost(cms::alpakatools::host(), sizes[0], sizes[1], sizes[5]);
      alpaka::memcpy(queue, outputHost.buffer(), genericCollection.buffer());
      alpaka::wait(queue);

      check(h_nestedBlocksCollection.const_view(), outputHost.const_view());
    }

    SECTION("Deep copy the ConstView device to host") {
      GenericSoAConstView genericConstView{nestedBlocksConstView.firstBlocks(), nestedBlocksConstView.secondLayout()};

      // PortableCollection that will host the aggregated columns
      PortableHostCollection<GenericSoA> genericCollection(queue, sizes[0], sizes[1], sizes[5]);
      genericCollection.deepCopy(queue, genericConstView);
      alpaka::wait(queue);

      GenericSoAConstView copiedGenericConstView = genericCollection.const_view();
      REQUIRE(copiedGenericConstView.blocks().first().metadata().addressOf_column() !=
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.blocks().first().metadata().addressOf_vector() !=
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_vector());
      REQUIRE(copiedGenericConstView.blocks().second().metadata().addressOf_column() !=
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.blocks().second().metadata().addressOf_vector() !=
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_vector());
      REQUIRE(copiedGenericConstView.layout().metadata().addressOf_column() !=
              nestedBlocksConstView.secondLayout().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.layout().metadata().addressOf_vector() !=
              nestedBlocksConstView.secondLayout().metadata().addressOf_vector());

      check(h_nestedBlocksCollection.const_view(), genericCollection.const_view());
    }

    SECTION("Deep copy the ConstView host to device") {
      GenericSoAConstView genericConstView{h_nestedBlocksCollection.const_view().firstBlocks(),
                                           h_nestedBlocksCollection.const_view().secondLayout()};

      // PortableCollection that will host the aggregated columns
      PortableCollection<Device, GenericSoA> genericCollection(queue, sizes[0], sizes[1], sizes[5]);
      genericCollection.deepCopy(queue, genericConstView);
      alpaka::wait(queue);

      GenericSoAConstView copiedGenericConstView = genericCollection.const_view();
      REQUIRE(copiedGenericConstView.blocks().first().metadata().addressOf_column() !=
              h_nestedBlocksCollection.const_view().firstBlocks().first().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.blocks().first().metadata().addressOf_vector() !=
              h_nestedBlocksCollection.const_view().firstBlocks().first().metadata().addressOf_vector());
      REQUIRE(copiedGenericConstView.blocks().second().metadata().addressOf_column() !=
              h_nestedBlocksCollection.const_view().firstBlocks().second().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.blocks().second().metadata().addressOf_vector() !=
              h_nestedBlocksCollection.const_view().firstBlocks().second().metadata().addressOf_vector());
      REQUIRE(copiedGenericConstView.layout().metadata().addressOf_column() !=
              h_nestedBlocksCollection.const_view().secondLayout().metadata().addressOf_column());
      REQUIRE(copiedGenericConstView.layout().metadata().addressOf_vector() !=
              h_nestedBlocksCollection.const_view().secondLayout().metadata().addressOf_vector());

      PortableHostCollection<GenericSoA> outputHost(cms::alpakatools::host(), sizes[0], sizes[1], sizes[5]);
      alpaka::memcpy(queue, outputHost.buffer(), genericCollection.buffer());
      alpaka::wait(queue);

      check(h_nestedBlocksCollection.const_view(), outputHost.const_view());
    }
  }
}
