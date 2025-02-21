#include <Eigen/Core>
#include <Eigen/Dense>

#include <catch.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

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

GENERATE_SOA_LAYOUT(CustomizedSoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using CustomizedSoA = CustomizedSoATemplate<>;
using CustomizedSoAView = CustomizedSoA::View;
using CustomizedSoAConstView = CustomizedSoA::ConstView;

TEST_CASE("Aggregate from SoA Customized View") {
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
    positionCollectionView.x()[i] = static_cast<float>(i);
    positionCollectionView.y()[i] = static_cast<float>(i) * 2.0f;
    positionCollectionView.z()[i] = static_cast<float>(i) * 3.0f;
  }
  positionCollectionView.detectorType() = 1;

  float time = 0.01;
  for (size_t i = 0; i < elems; i++) {
    pcaCollectionView.eigenvector_1()[i] = positionCollectionView.x()[i] / time;
    pcaCollectionView.eigenvector_2()[i] = positionCollectionView.y()[i] / time;
    pcaCollectionView.eigenvector_3()[i] = positionCollectionView.z()[i] / time;
    pcaCollectionView[i].candidateDirection()(0) = positionCollectionView[i].x() / time;
    pcaCollectionView[i].candidateDirection()(1) = positionCollectionView[i].y() / time;
    pcaCollectionView[i].candidateDirection()(2) = positionCollectionView[i].z() / time;
  }

  SECTION("Aggregate the View") {
    // addresses and size of the SoA columns
    const auto posRecs = positionCollectionView.records();
    const auto pcaRecs = pcaCollectionView.records();

    // building the View with runtime check for the size
    CustomizedSoAView customView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for equality of memory addresses
    REQUIRE(customView.metadata().addressOf_x() == positionCollectionView.metadata().addressOf_x());
    REQUIRE(customView.metadata().addressOf_y() == positionCollectionView.metadata().addressOf_y());
    REQUIRE(customView.metadata().addressOf_z() == positionCollectionView.metadata().addressOf_z());
    REQUIRE(customView.metadata().addressOf_candidateDirection() ==
            pcaCollectionView.metadata().addressOf_candidateDirection());

    // PortableHostCollection that will host the aggregated columns
    PortableHostCollection<CustomizedSoA> customCollection(elems, cms::alpakatools::host());
    customCollection.aggregate(customView);

    // Check for inequality of memory addresses
    REQUIRE(customCollection.view().metadata().addressOf_x() != positionCollectionView.metadata().addressOf_x());
    REQUIRE(customCollection.view().metadata().addressOf_y() != positionCollectionView.metadata().addressOf_y());
    REQUIRE(customCollection.view().metadata().addressOf_z() != positionCollectionView.metadata().addressOf_z());
    REQUIRE(customCollection.view().metadata().addressOf_candidateDirection() !=
            pcaCollectionView.metadata().addressOf_candidateDirection());
  }

  SECTION("Aggregate the ConstView") {
    // addresses and size of the SoA columns
    const auto posRecs = positionCollectionConstView.records();
    const auto pcaRecs = pcaCollectionConstView.records();

    // building the View with runtime check for the size
    CustomizedSoAConstView customConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for equality of memory addresses
    REQUIRE(customConstView.metadata().addressOf_x() == positionCollectionView.metadata().addressOf_x());
    REQUIRE(customConstView.metadata().addressOf_y() == positionCollectionView.metadata().addressOf_y());
    REQUIRE(customConstView.metadata().addressOf_z() == positionCollectionView.metadata().addressOf_z());
    REQUIRE(customConstView.metadata().addressOf_candidateDirection() ==
            pcaCollectionView.metadata().addressOf_candidateDirection());

    // PortableHostCollection that will host the aggregated columns
    PortableHostCollection<CustomizedSoA> customCollection(elems, cms::alpakatools::host());
    customCollection.aggregate(customConstView);

    // Check for inequality of memory addresses
    REQUIRE(customCollection.view().metadata().addressOf_x() != positionCollectionView.metadata().addressOf_x());
    REQUIRE(customCollection.view().metadata().addressOf_y() != positionCollectionView.metadata().addressOf_y());
    REQUIRE(customCollection.view().metadata().addressOf_z() != positionCollectionView.metadata().addressOf_z());
    REQUIRE(customCollection.view().metadata().addressOf_candidateDirection() !=
            pcaCollectionView.metadata().addressOf_candidateDirection());
  }
}