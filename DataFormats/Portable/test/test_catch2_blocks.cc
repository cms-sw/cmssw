#include <Eigen/Core>
#include <Eigen/Dense>

#include <catch.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, eigenvector_1),
                    SOA_COLUMN(float, eigenvector_2),
                    SOA_COLUMN(float, eigenvector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;

GENERATE_SOA_BLOCKS(SoAGenericBlocksTemplate, SOA_BLOCK(position, SoAPositionTemplate), SOA_BLOCK(pca, SoAPCATemplate))

using SoAGenericBlocks = SoAGenericBlocksTemplate<>;
using SoAGenericBlocksView = SoAGenericBlocks::View;
using SoAGenericBlocksConstView = SoAGenericBlocks::ConstView;

TEST_CASE("Deep copy from SoABlocks Generic View") {
  // different number of elements for the SoAs
  const std::size_t elemsPos = 10;
  const std::size_t elemsPCA = 20;

  // Portable Collections
  PortableHostCollection<SoAPosition> positionCollection(elemsPos, cms::alpakatools::host());
  PortableHostCollection<SoAPCA> pcaCollection(elemsPCA, cms::alpakatools::host());

  // Portable Collection Views
  SoAPositionView& positionCollectionView = positionCollection.view();
  SoAPCAView& pcaCollectionView = pcaCollection.view();
  // Portable Collection ConstViews
  const SoAPositionConstView& positionCollectionConstView = positionCollection.const_view();
  const SoAPCAConstView& pcaCollectionConstView = pcaCollection.const_view();

  // fill up
  for (size_t i = 0; i < elemsPos; i++) {
    positionCollectionView[i] = {i * 1.0f, i * 2.0f, i * 3.0f};
  }
  positionCollectionView.detectorType() = 1;

  float time = 0.01;
  for (size_t i = 0; i < elemsPCA; i++) {
    pcaCollectionView[i].eigenvector_1() = (i * 1.0f) / time;
    pcaCollectionView[i].eigenvector_2() = (i * 2.0f) / time;
    pcaCollectionView[i].eigenvector_3() = (i * 3.0f) / time;
    pcaCollectionView[i].candidateDirection()(0) = (i * 1.0) / time;
    pcaCollectionView[i].candidateDirection()(1) = (i * 2.0) / time;
    pcaCollectionView[i].candidateDirection()(2) = (i * 3.0) / time;
  }

  SECTION("Deep copy the BlocksView") {
    // building the SoABlocks View, there is no need for runtime check for the size since they are different
    SoAGenericBlocksView genericBlocksView{positionCollectionView, pcaCollectionView};

    // Check for equality of memory addresses
    REQUIRE(genericBlocksView.position().metadata().addressOf_x() == positionCollectionView.metadata().addressOf_x());
    REQUIRE(genericBlocksView.position().metadata().addressOf_y() == positionCollectionView.metadata().addressOf_y());
    REQUIRE(genericBlocksView.position().metadata().addressOf_z() == positionCollectionView.metadata().addressOf_z());
    REQUIRE(genericBlocksView.pca().metadata().addressOf_candidateDirection() ==
            pcaCollectionView.metadata().addressOf_candidateDirection());

    // PortableHostCollection that will host the aggregated columns
    std::array<cms::soa::size_type, 2> sizes{{elemsPos, elemsPCA}};
    PortableHostCollection<SoAGenericBlocks> genericCollection(sizes, cms::alpakatools::host());
    genericCollection.deepCopy(genericBlocksView);

    // Check for inequality of memory addresses
    REQUIRE(genericCollection.view().position().metadata().addressOf_x() !=
            positionCollectionView.metadata().addressOf_x());
    REQUIRE(genericCollection.view().position().metadata().addressOf_y() !=
            positionCollectionView.metadata().addressOf_y());
    REQUIRE(genericCollection.view().position().metadata().addressOf_z() !=
            positionCollectionView.metadata().addressOf_z());
    REQUIRE(genericCollection.view().pca().metadata().addressOf_candidateDirection() !=
            pcaCollectionView.metadata().addressOf_candidateDirection());
  }

  SECTION("Deep copy the BlocksConstView") {
    // building the SoABlocks View, there is no need for runtime check for the size since they are different
    SoAGenericBlocksConstView genericBlocksConstView{positionCollectionConstView, pcaCollectionConstView};

    // Check for equality of memory addresses
    REQUIRE(genericBlocksConstView.position().metadata().addressOf_x() ==
            positionCollectionConstView.metadata().addressOf_x());
    REQUIRE(genericBlocksConstView.position().metadata().addressOf_y() ==
            positionCollectionConstView.metadata().addressOf_y());
    REQUIRE(genericBlocksConstView.position().metadata().addressOf_z() ==
            positionCollectionConstView.metadata().addressOf_z());
    REQUIRE(genericBlocksConstView.pca().metadata().addressOf_candidateDirection() ==
            pcaCollectionConstView.metadata().addressOf_candidateDirection());

    // PortableHostCollection that will host the aggregated columns
    std::array<cms::soa::size_type, 2> sizes{{elemsPos, elemsPCA}};
    PortableHostCollection<SoAGenericBlocks> genericCollection(sizes, cms::alpakatools::host());
    genericCollection.deepCopy(genericBlocksConstView);

    // Check for inequality of memory addresses
    REQUIRE(genericCollection.const_view().position().metadata().addressOf_x() !=
            positionCollectionConstView.metadata().addressOf_x());
    REQUIRE(genericCollection.const_view().position().metadata().addressOf_y() !=
            positionCollectionConstView.metadata().addressOf_y());
    REQUIRE(genericCollection.const_view().position().metadata().addressOf_z() !=
            positionCollectionConstView.metadata().addressOf_z());
    REQUIRE(genericCollection.const_view().pca().metadata().addressOf_candidateDirection() !=
            pcaCollectionConstView.metadata().addressOf_candidateDirection());
  }
}
