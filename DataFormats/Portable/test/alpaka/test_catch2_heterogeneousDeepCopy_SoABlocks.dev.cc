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

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

constexpr float step = 0.01;

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;

GENERATE_SOA_BLOCKS(SoAGenericBlocksTemplate, SOA_BLOCK(position, SoAPositionTemplate), SOA_BLOCK(pca, SoAPCATemplate))

using SoAGenericBlocks = SoAGenericBlocksTemplate<>;
using SoAGenericBlocksView = SoAGenericBlocks::View;
using SoAGenericBlocksConstView = SoAGenericBlocks::ConstView;

// Fill SoAs
struct FillSoAPosition {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, SoAPositionView positionView) const {
    if (cms::alpakatools::once_per_grid(acc))
      positionView.detectorType() = 1;

    for (auto local_idx : cms::alpakatools::uniform_elements(acc, positionView.metadata().size())) {
      positionView[local_idx].x() = static_cast<float>(local_idx);
      positionView[local_idx].y() = static_cast<float>(local_idx) * 2.0f;
      positionView[local_idx].z() = static_cast<float>(local_idx) * 3.0f;
    }
  }
};

struct FillSoAPCA {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, SoAPCAView pcaView) const {
    for (auto local_idx : cms::alpakatools::uniform_elements(acc, pcaView.metadata().size())) {
      pcaView[local_idx].vector_1() = 1.0f / step;
      pcaView[local_idx].vector_2() = 2.0f / step;
      pcaView[local_idx].vector_3() = 3.0f / step;
      pcaView[local_idx].candidateDirection()(0) = 1.0f / step;
      pcaView[local_idx].candidateDirection()(1) = 2.0f / step;
      pcaView[local_idx].candidateDirection()(2) = 3.0f / step;
    }
  }
};

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

    // Number of elements
    const int elemPos = 10;
    const int elemPCA = 100;

    PortableCollection<Device, SoAPosition> positionCollection(queue, elemPos);
    PortableCollection<Device, SoAPCA> pcaCollection(queue, elemPCA);

    // Portable Collection Views
    SoAPositionView& positionCollectionView = positionCollection.view();
    SoAPCAView& pcaCollectionView = pcaCollection.view();

    // fill up
    auto blockSize = 64;
    auto numberOfBlocks = cms::alpakatools::divide_up_by(elemPos, blockSize);

    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDiv, FillSoAPosition{}, positionCollectionView);

    alpaka::wait(queue);

    numberOfBlocks = cms::alpakatools::divide_up_by(elemPCA, blockSize);

    const auto workDivPCA = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDivPCA, FillSoAPCA{}, pcaCollectionView);

    alpaka::wait(queue);

    // Build the View of the SoABlocks
    SoAGenericBlocksView genericBlocksView{positionCollectionView, pcaCollectionView};

    SECTION("Heterogeneous deep copy of the SoABlocks View") {
      // PortableCollection that will host the aggregated columns
      PortableCollection<Device, SoAGenericBlocks> genericCollection(queue, elemPos, elemPCA);
      genericCollection.deepCopy(queue, genericBlocksView);
    }
  }
}
