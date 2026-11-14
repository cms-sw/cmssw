#include <catch2/catch_all.hpp>

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "RecoEgamma/EgammaTools/interface/HGCalShowerShapeHelper.h"

#include <memory>
#include <unordered_map>
#include <vector>

TEST_CASE("HGCalShowerShapeHelper cell size per silicon wafer type", "[HGCalShowerShapeHelper]") {
  const HGCalShowerShapeHelper::ShowerShapeCalc calc(
      std::make_shared<const ticlgeom::Tools>(),
      std::make_shared<const std::unordered_map<uint32_t, const reco::PFRecHit*>>(),
      std::vector<std::pair<DetId, float>>(),
      1.0,
      0.,
      0.,
      1,
      26);
  auto cellSize = [&calc](int type) {
    return calc.getCellSize(HGCSiliconDetId(DetId::HGCalEE, 1, type, 10, 3, 0, 5, 5));
  };

  SECTION("High-density wafers get the high-density cell size") {
    CHECK(cellSize(HGCSiliconDetId::HGCalHD120) == HGCalShowerShapeHelper::kHDWaferCellSize_);
    CHECK(cellSize(HGCSiliconDetId::HGCalHD200) == HGCalShowerShapeHelper::kHDWaferCellSize_);
  }
  SECTION("Low-density wafers get the low-density cell size") {
    CHECK(cellSize(HGCSiliconDetId::HGCalLD200) == HGCalShowerShapeHelper::kLDWaferCellSize_);
    CHECK(cellSize(HGCSiliconDetId::HGCalLD300) == HGCalShowerShapeHelper::kLDWaferCellSize_);
  }
}
