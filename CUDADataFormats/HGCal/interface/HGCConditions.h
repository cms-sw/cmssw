#ifndef CUDADataFormats_HGCal_HGCConditions_h
#define CUDADataFormats_HGCal_HGCConditions_h

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>

class HeterogeneousHGCSiliconDetId {
public:
  constexpr HeterogeneousHGCSiliconDetId(uint32_t id) : id_(id) {}
  constexpr std::int32_t type() { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }
  constexpr std::int32_t zside() { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }
  constexpr std::int32_t layer() { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }
  constexpr std::int32_t waferUAbs() { return (id_ >> kHGCalWaferUOffset) & kHGCalWaferUMask; }
  constexpr std::int32_t waferVAbs() { return (id_ >> kHGCalWaferVOffset) & kHGCalWaferVMask; }
  constexpr std::int32_t waferU() {
    return (((id_ >> kHGCalWaferUSignOffset) & kHGCalWaferUSignMask) ? -waferUAbs() : waferUAbs());
  }
  constexpr std::int32_t waferV() {
    return (((id_ >> kHGCalWaferVSignOffset) & kHGCalWaferVSignMask) ? -waferVAbs() : waferVAbs());
  }
  constexpr std::int32_t waferX() { return (-2 * waferU() + waferV()); }
  constexpr std::int32_t waferY() { return (2 * waferV()); }
  constexpr std::int32_t cellU() { return (id_ >> kHGCalCellUOffset) & kHGCalCellUMask; }
  constexpr std::int32_t cellV() { return (id_ >> kHGCalCellVOffset) & kHGCalCellVMask; }
  constexpr std::int32_t nCellsSide() { return (type() == HGCalFine) ? HGCalFineN : HGCalCoarseN; }
  constexpr std::int32_t cellX() {
    const std::int32_t N = nCellsSide();
    return (3 * (cellV() - N) + 2);
  }
  constexpr std::int32_t cellY() {
    const std::int32_t N = nCellsSide();
    return (2 * cellU() - (N + cellV()));
  }

private:
  std::uint32_t id_;
  enum waferType { HGCalFine = 0, HGCalCoarseThin = 1, HGCalCoarseThick = 2 };
  static constexpr std::int32_t HGCalFineN = 12;
  static constexpr std::int32_t HGCalCoarseN = 8;
  static constexpr std::int32_t kHGCalCellUOffset = 0;
  static constexpr std::int32_t kHGCalCellUMask = 0x1F;
  static constexpr std::int32_t kHGCalCellVOffset = 5;
  static constexpr std::int32_t kHGCalCellVMask = 0x1F;
  static constexpr std::int32_t kHGCalWaferUOffset = 10;
  static constexpr std::int32_t kHGCalWaferUMask = 0xF;
  static constexpr std::int32_t kHGCalWaferUSignOffset = 14;
  static constexpr std::int32_t kHGCalWaferUSignMask = 0x1;
  static constexpr std::int32_t kHGCalWaferVOffset = 15;
  static constexpr std::int32_t kHGCalWaferVMask = 0xF;
  static constexpr std::int32_t kHGCalWaferVSignOffset = 19;
  static constexpr std::int32_t kHGCalWaferVSignMask = 0x1;
  static constexpr std::int32_t kHGCalLayerOffset = 20;
  static constexpr std::int32_t kHGCalLayerMask = 0x1F;
  static constexpr std::int32_t kHGCalZsideOffset = 25;
  static constexpr std::int32_t kHGCalZsideMask = 0x1;
  static constexpr std::int32_t kHGCalTypeOffset = 26;
  static constexpr std::int32_t kHGCalTypeMask = 0x3;
};

class HeterogeneousHGCScintillatorDetId {
public:
  constexpr HeterogeneousHGCScintillatorDetId(uint32_t id) : id_(id) {}
  constexpr std::int32_t type() { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }
  constexpr std::int32_t zside() const { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }
  constexpr std::int32_t layer() const { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }

private:
  std::uint32_t id_;
  static constexpr std::uint32_t kHGCalPhiOffset = 0;
  static constexpr std::uint32_t kHGCalPhiMask = 0x1FF;
  static constexpr std::uint32_t kHGCalRadiusOffset = 9;
  static constexpr std::uint32_t kHGCalRadiusMask = 0xFF;
  static constexpr std::uint32_t kHGCalLayerOffset = 17;
  static constexpr std::uint32_t kHGCalLayerMask = 0x1F;
  static constexpr std::uint32_t kHGCalTriggerOffset = 22;
  static constexpr std::uint32_t kHGCalTriggerMask = 0x1;
  static constexpr std::uint32_t kHGCalZsideOffset = 25;
  static constexpr std::uint32_t kHGCalZsideMask = 0x1;
  static constexpr std::uint32_t kHGCalTypeOffset = 26;
  static constexpr std::uint32_t kHGCalTypeMask = 0x3;
};

namespace hgcal_conditions {
  namespace parameters {
    enum class HeterogeneousHGCalEEParametersType { Double, Int32_t };
    enum class HeterogeneousHGCalHEFParametersType { Double, Int32_t };
    enum class HeterogeneousHGCalHEBParametersType { Double, Int32_t };

    const std::array<HeterogeneousHGCalEEParametersType, 5> typesEE = {{HeterogeneousHGCalEEParametersType::Double,
                                                                        HeterogeneousHGCalEEParametersType::Double,
                                                                        HeterogeneousHGCalEEParametersType::Double,
                                                                        HeterogeneousHGCalEEParametersType::Double,
                                                                        HeterogeneousHGCalEEParametersType::Int32_t}};

    const std::array<HeterogeneousHGCalHEFParametersType, 5> typesHEF = {
        {HeterogeneousHGCalHEFParametersType::Double,
         HeterogeneousHGCalHEFParametersType::Double,
         HeterogeneousHGCalHEFParametersType::Double,
         HeterogeneousHGCalHEFParametersType::Double,
         HeterogeneousHGCalHEFParametersType::Int32_t}};

    const std::array<HeterogeneousHGCalHEBParametersType, 2> typesHEB = {
        {HeterogeneousHGCalHEBParametersType::Double, HeterogeneousHGCalHEBParametersType::Int32_t}};

    class HeterogeneousHGCalEEParameters {
    public:
      //indexed by cell number
      double *cellFineX_;
      double *cellFineY_;
      double *cellCoarseX_;
      double *cellCoarseY_;
      //index by wafer number
      std::int32_t *waferTypeL_;
    };
    class HeterogeneousHGCalHEFParameters {
    public:
      //indexed by cell number
      double *cellFineX_;
      double *cellFineY_;
      double *cellCoarseX_;
      double *cellCoarseY_;
      //index by wafer number
      std::int32_t *waferTypeL_;
    };
    class HeterogeneousHGCalHEBParameters {
    public:
      double *testD_;
      std::int32_t *testI_;
    };

  }  //namespace parameters

  namespace positions {

    enum class HeterogeneousHGCalPositionsType { Float, Int32_t, Uint32_t };

    const std::vector<HeterogeneousHGCalPositionsType> types = {HeterogeneousHGCalPositionsType::Float,
                                                                HeterogeneousHGCalPositionsType::Float,
                                                                HeterogeneousHGCalPositionsType::Float,
                                                                HeterogeneousHGCalPositionsType::Int32_t,
                                                                HeterogeneousHGCalPositionsType::Int32_t,
                                                                HeterogeneousHGCalPositionsType::Int32_t,
                                                                HeterogeneousHGCalPositionsType::Uint32_t};

    struct HGCalPositionsMapping {
      std::vector<float> zLayer;                    //z position per layer
      std::vector<std::int32_t> nCellsLayer;        //#cells per layer
      std::vector<std::int32_t> nCellsWaferUChunk;  //#cells per U wafer (each in turn including all V wafers)
      std::vector<std::int32_t> nCellsHexagon;      //#cells per V wafer
      std::vector<std::uint32_t> detid;
      //variables required for calculating the positions (x,y) from the detid in the GPU
      float waferSize;
      float sensorSeparation;
      //variables required for the mapping of detid -> cell in the geometry
      std::int32_t firstLayer;
      std::int32_t lastLayer;
      std::int32_t waferMax;
      std::int32_t waferMin;
    };

    struct HeterogeneousHGCalPositionsMapping {
      //the x, y and z positions will not be filled in the CPU
      float *x;
      float *y;
      float *zLayer;
      std::int32_t *nCellsLayer;
      std::int32_t *nCellsWaferUChunk;
      std::int32_t *nCellsHexagon;
      std::uint32_t *detid;
      //variables required for calculating the positions (x,y) from the detid in the GPU
      float waferSize;
      float sensorSeparation;
      //variables required for the mapping of detid -> cell in the geometry
      std::int32_t firstLayer;
      std::int32_t lastLayer;
      std::int32_t waferMax;
      std::int32_t waferMin;
    };

  }  //namespace positions

  struct HeterogeneousEEConditionsESProduct {
    parameters::HeterogeneousHGCalEEParameters params;
  };
  struct HeterogeneousHEFConditionsESProduct {
    parameters::HeterogeneousHGCalHEFParameters params;
    //positions::HeterogeneousHGCalPositionsMapping posmap;
    //size_t nelems_posmap;
  };
  struct HeterogeneousHEBConditionsESProduct {
    parameters::HeterogeneousHGCalHEBParameters params;
  };

  struct HeterogeneousHEFCellPositionsConditionsESProduct {
    positions::HeterogeneousHGCalPositionsMapping posmap;
    std::size_t nelems_posmap;
  };

}  // namespace hgcal_conditions

#endif  //CUDADataFormats_HGCal_HGCConditions_h
