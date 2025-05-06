#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

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

using GenericSoA = GenericSoATemplate<cms::soa::CacheLineSize::IntelCPU>;
using GenericSoAView = GenericSoA::View;
using GenericSoAConstView = GenericSoA::ConstView;

TEST_CASE("SoAGenericView") {
  // common number of elements for the SoAs
  const std::size_t elems = 17;

  // buffer sizes
  const std::size_t positionBufferSize = SoAPosition::computeDataSize(elems);
  const std::size_t pcaBufferSize = SoAPCA::computeDataSize(elems);

  // memory buffer for the SoA of positions
  std::unique_ptr<std::byte, decltype(std::free) *> bufferPos{
      reinterpret_cast<std::byte *>(aligned_alloc(SoAPosition::alignment, positionBufferSize)), std::free};
  // memory buffer for the SoA of the PCA
  std::unique_ptr<std::byte, decltype(std::free) *> bufferPCA{
      reinterpret_cast<std::byte *>(aligned_alloc(SoAPCA::alignment, pcaBufferSize)), std::free};

  // SoA Layouts
  SoAPosition position{bufferPos.get(), elems};
  SoAPCA pca{bufferPCA.get(), elems};

  // SoA Views
  SoAPositionView positionView{position};
  SoAPositionConstView positionConstView{position};
  SoAPCAView pcaView{pca};
  SoAPCAConstView pcaConstView{pca};

  // fill up
  for (size_t i = 0; i < elems; i++) {
    positionView[i] = {i * 1.0f, i * 2.0f, i * 3.0f};
  }
  positionView.detectorType() = 1;

  float time = 0.01;
  for (size_t i = 0; i < elems; i++) {
    pcaView[i].eigenvector_1() = positionView[i].x() / time;
    pcaView[i].eigenvector_2() = positionView[i].y() / time;
    pcaView[i].eigenvector_3() = positionView[i].z() / time;
    pcaView[i].candidateDirection()(0) = positionView[i].x() / time;
    pcaView[i].candidateDirection()(1) = positionView[i].y() / time;
    pcaView[i].candidateDirection()(2) = positionView[i].z() / time;
  }

  SECTION("Generic View") {
    // addresses and size of the SoA columns
    const auto posRecs = positionView.records();
    const auto pcaRecs = pcaView.records();

    // building the View with runtime check for the size
    GenericSoAView genericView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for equality of memory addresses
    REQUIRE(genericView.metadata().addressOf_xPos() == positionView.metadata().addressOf_x());
    REQUIRE(genericView.metadata().addressOf_yPos() == positionView.metadata().addressOf_y());
    REQUIRE(genericView.metadata().addressOf_zPos() == positionView.metadata().addressOf_z());
    REQUIRE(genericView.metadata().addressOf_candidateDirection() == pcaView.metadata().addressOf_candidateDirection());

    // Check for reference to original SoA
    genericView[3].xPos() = 0.;
    REQUIRE(genericView[3].xPos() == positionConstView[3].x());
  }

  SECTION("Generic ConstView") {
    // addresses and size of the SoA columns
    const auto posRecs = positionConstView.records();
    const auto pcaRecs = pcaConstView.records();

    // building the ConstView with runtime check for the size
    GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for equality of memory addresses
    REQUIRE(genericConstView.metadata().addressOf_xPos() == positionConstView.metadata().addressOf_x());
    REQUIRE(genericConstView.metadata().addressOf_yPos() == positionConstView.metadata().addressOf_y());
    REQUIRE(genericConstView.metadata().addressOf_zPos() == positionConstView.metadata().addressOf_z());
    REQUIRE(genericConstView.metadata().addressOf_candidateDirection() ==
            pcaConstView.metadata().addressOf_candidateDirection());
  }

  SECTION("Generic ConstView from Views") {
    // addresses and size of the SoA columns
    const auto posRecs = positionView.records();
    const auto pcaRecs = pcaView.records();

    // building the ConstView with runtime check for the size
    GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // Check for reference to the Generic SoA - it is possible to modify the ConstView by reference modifying the Views
    positionView[3].x() = 0.;
    REQUIRE(genericConstView[3].xPos() == positionView[3].x());
  }

  SECTION("Deep copy the Generic View") {
    // building the Layout
    const std::size_t genericBufferSize = GenericSoA::computeDataSize(elems);
    std::unique_ptr<std::byte, decltype(std::free) *> bufferGeneric{
        reinterpret_cast<std::byte *>(aligned_alloc(GenericSoA::alignment, genericBufferSize)), std::free};
    GenericSoA genericSoA(bufferGeneric.get(), elems);

    // building the Generic View
    const auto posRecs = positionView.records();
    const auto pcaRecs = pcaView.records();
    GenericSoAView genericView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    // aggregate the columns from the view with runtime check for the size
    genericSoA.deepCopy(genericView);

    // building the View of the new SoA
    GenericSoAView genericSoAView{genericSoA};

    // Check for inequality of memory addresses
    REQUIRE(genericSoAView.metadata().addressOf_xPos() != positionConstView.metadata().addressOf_x());
    REQUIRE(genericSoAView.metadata().addressOf_yPos() != positionConstView.metadata().addressOf_y());
    REQUIRE(genericSoAView.metadata().addressOf_zPos() != positionConstView.metadata().addressOf_z());
    REQUIRE(genericSoAView.metadata().addressOf_candidateDirection() !=
            pcaConstView.metadata().addressOf_candidateDirection());

    // Check for column alignments
    REQUIRE(0 ==
            reinterpret_cast<uintptr_t>(genericSoAView.metadata().addressOf_xPos()) % decltype(genericSoA)::alignment);
    REQUIRE(0 ==
            reinterpret_cast<uintptr_t>(genericSoAView.metadata().addressOf_yPos()) % decltype(genericSoA)::alignment);
    REQUIRE(0 ==
            reinterpret_cast<uintptr_t>(genericSoAView.metadata().addressOf_zPos()) % decltype(genericSoA)::alignment);
    REQUIRE(0 == reinterpret_cast<uintptr_t>(genericSoAView.metadata().addressOf_candidateDirection()) %
                     decltype(genericSoA)::alignment);

    // Check for contiguity of columns
    REQUIRE(reinterpret_cast<std::byte *>(genericSoAView.metadata().addressOf_xPos()) +
                cms::soa::alignSize(elems * sizeof(float), GenericSoA::alignment) ==
            reinterpret_cast<std::byte *>(genericSoAView.metadata().addressOf_yPos()));
    REQUIRE(reinterpret_cast<std::byte *>(genericSoAView.metadata().addressOf_yPos()) +
                cms::soa::alignSize(elems * sizeof(float), GenericSoA::alignment) ==
            reinterpret_cast<std::byte *>(genericSoAView.metadata().addressOf_zPos()));
    REQUIRE(reinterpret_cast<std::byte *>(genericSoAView.metadata().addressOf_zPos()) +
                cms::soa::alignSize(elems * sizeof(float), GenericSoA::alignment) ==
            reinterpret_cast<std::byte *>(genericSoAView.metadata().addressOf_candidateDirection()));

    // Ckeck the correctness of the copy
    for (size_t i = 0; i < elems; i++) {
      REQUIRE(genericSoAView[i].xPos() == positionConstView[i].x());
      REQUIRE(genericSoAView[i].yPos() == positionConstView[i].y());
      REQUIRE(genericSoAView[i].zPos() == positionConstView[i].z());
      REQUIRE(genericSoAView[i].candidateDirection()(0) == pcaConstView[i].candidateDirection()(0));
      REQUIRE(genericSoAView[i].candidateDirection()(1) == pcaConstView[i].candidateDirection()(1));
      REQUIRE(genericSoAView[i].candidateDirection()(2) == pcaConstView[i].candidateDirection()(2));
    }

    // Check for the independency of the aggregated SoA
    genericSoAView[3].xPos() = 0.;
    REQUIRE(genericSoAView[3].xPos() != positionView[3].x());
  }
}
