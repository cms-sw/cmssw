#include <Eigen/Core>
#include <Eigen/Dense>

#include <catch.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

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

GENERATE_SOA_LAYOUT(GenericSoATemplate,
                    SOA_COLUMN(float, xPos),
                    SOA_COLUMN(float, yPos),
                    SOA_COLUMN(float, zPos),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using GenericSoA = GenericSoATemplate<>;
using GenericSoAView = GenericSoA::View;
using GenericSoAConstView = GenericSoA::ConstView;

TEST_CASE("Deep copy from SoA Generic View") {
  // common number of elements for the SoAs
  const std::size_t elems = 10;

  // Portable Collections
  PortableHostCollection<SoAPosition> positionCollection(elems, cms::alpakatools::host());
  PortableHostCollection<SoAPCA> pcaCollection(elems, cms::alpakatools::host());

  // Portable Collection Views
  SoAPositionView& positionCollectionView = positionCollection.view();
  SoAPCAView& pcaCollectionView = pcaCollection.view();
  // Portable Collection ConstViews
  const SoAPositionConstView& positionCollectionConstView = positionCollection.const_view();
  const SoAPCAConstView& pcaCollectionConstView = pcaCollection.const_view();

  // fill up
  for (size_t i = 0; i < elems; i++) {
    positionCollectionView[i] = {i * 1.0f, i * 2.0f, i * 3.0f};
  }
  positionCollectionView.detectorType() = 1;

  float time = 0.01;
  for (size_t i = 0; i < elems; i++) {
    pcaCollectionView[i].eigenvector_1() = positionCollectionView[i].x() / time;
    pcaCollectionView[i].eigenvector_2() = positionCollectionView[i].y() / time;
    pcaCollectionView[i].eigenvector_3() = positionCollectionView[i].z() / time;
    pcaCollectionView[i].candidateDirection()(0) = positionCollectionView[i].x() / time;
    pcaCollectionView[i].candidateDirection()(1) = positionCollectionView[i].y() / time;
    pcaCollectionView[i].candidateDirection()(2) = positionCollectionView[i].z() / time;
  }

  SECTION("Deep copy the View") {
    // addresses and size of the SoA columns
    const auto posRecs = positionCollectionView.records();
    const auto pcaRecs = pcaCollectionView.records();

    // building the View with runtime check for the size
    GenericSoAView genericView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for equality of memory addresses
    REQUIRE(genericView.metadata().addressOf_xPos() == positionCollectionView.metadata().addressOf_x());
    REQUIRE(genericView.metadata().addressOf_yPos() == positionCollectionView.metadata().addressOf_y());
    REQUIRE(genericView.metadata().addressOf_zPos() == positionCollectionView.metadata().addressOf_z());
    REQUIRE(genericView.metadata().addressOf_candidateDirection() ==
            pcaCollectionView.metadata().addressOf_candidateDirection());

    // PortableHostCollection that will host the aggregated columns
    PortableHostCollection<GenericSoA> genericCollection(elems, cms::alpakatools::host());
    genericCollection.deepCopy(genericView);

    // Check for inequality of memory addresses
    REQUIRE(genericCollection.view().metadata().addressOf_xPos() != positionCollectionView.metadata().addressOf_x());
    REQUIRE(genericCollection.view().metadata().addressOf_yPos() != positionCollectionView.metadata().addressOf_y());
    REQUIRE(genericCollection.view().metadata().addressOf_zPos() != positionCollectionView.metadata().addressOf_z());
    REQUIRE(genericCollection.view().metadata().addressOf_candidateDirection() !=
            pcaCollectionView.metadata().addressOf_candidateDirection());
  }

  SECTION("Deep copy the ConstView") {
    // addresses and size of the SoA columns
    const auto posRecs = positionCollectionConstView.records();
    const auto pcaRecs = pcaCollectionConstView.records();

    // building the View with runtime check for the size
    GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for equality of memory addresses
    REQUIRE(genericConstView.metadata().addressOf_xPos() == positionCollectionView.metadata().addressOf_x());
    REQUIRE(genericConstView.metadata().addressOf_yPos() == positionCollectionView.metadata().addressOf_y());
    REQUIRE(genericConstView.metadata().addressOf_zPos() == positionCollectionView.metadata().addressOf_z());
    REQUIRE(genericConstView.metadata().addressOf_candidateDirection() ==
            pcaCollectionView.metadata().addressOf_candidateDirection());

    // PortableHostCollection that will host the aggregated columns
    PortableHostCollection<GenericSoA> genericCollection(elems, cms::alpakatools::host());
    genericCollection.deepCopy(genericConstView);

    // Check for inequality of memory addresses
    REQUIRE(genericCollection.view().metadata().addressOf_xPos() != positionCollectionView.metadata().addressOf_x());
    REQUIRE(genericCollection.view().metadata().addressOf_yPos() != positionCollectionView.metadata().addressOf_y());
    REQUIRE(genericCollection.view().metadata().addressOf_zPos() != positionCollectionView.metadata().addressOf_z());
    REQUIRE(genericCollection.view().metadata().addressOf_candidateDirection() !=
            pcaCollectionView.metadata().addressOf_candidateDirection());
  }
}
