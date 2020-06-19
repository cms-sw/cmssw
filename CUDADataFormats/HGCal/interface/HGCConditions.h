#ifndef CUDADataFormats_HGCal_HGCConditions_h
#define CUDADataFormats_HGCal_HGCConditions_h

class HeterogeneousHGCSiliconDetId {
 public:
  constexpr HeterogeneousHGCSiliconDetId(uint32_t id): id_(id) {}
  constexpr uint32_t type()     { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }
  constexpr int32_t zside()     { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }
  constexpr uint32_t layer()    { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }
  constexpr int32_t waferUAbs() { return (id_ >> kHGCalWaferUOffset) & kHGCalWaferUMask; }
  constexpr int32_t waferVAbs() { return (id_ >> kHGCalWaferVOffset) & kHGCalWaferVMask; }
  constexpr int32_t waferU()    { return (((id_ >> kHGCalWaferUSignOffset) & kHGCalWaferUSignMask) ? -waferUAbs() : waferUAbs()); }
  constexpr int32_t waferV()    { return (((id_ >> kHGCalWaferVSignOffset) & kHGCalWaferVSignMask) ? -waferVAbs() : waferVAbs()); }
  constexpr int32_t waferX()    { return (-2 * waferU() + waferV()); }
  constexpr int32_t waferY()    { return (2 * waferV()); }
  constexpr uint32_t cellU()    { return (id_ >> kHGCalCellUOffset) & kHGCalCellUMask; }
  constexpr uint32_t cellV()    { return (id_ >> kHGCalCellVOffset) & kHGCalCellVMask; }
  constexpr uint32_t nCells()   { return (type() == HGCalFine) ? HGCalFineN : HGCalCoarseN; }
  constexpr int32_t cellX()     { const uint32_t N = nCells(); return (3 * (cellV() - N) + 2); }
  constexpr int32_t cellY()     { const uint32_t N = nCells(); return (2 * cellU() - (N + cellV())); }

 private:
  uint32_t id_;
  enum waferType { HGCalFine = 0, HGCalCoarseThin = 1, HGCalCoarseThick = 2 };
  static const int32_t HGCalFineN = 12;
  static const int32_t HGCalCoarseN = 8;
  static const int32_t kHGCalCellUOffset = 0;
  static const int32_t kHGCalCellUMask = 0x1F;
  static const int32_t kHGCalCellVOffset = 5;
  static const int32_t kHGCalCellVMask = 0x1F;
  static const int32_t kHGCalWaferUOffset = 10;
  static const int32_t kHGCalWaferUMask = 0xF;
  static const int32_t kHGCalWaferUSignOffset = 14;
  static const int32_t kHGCalWaferUSignMask = 0x1;
  static const int32_t kHGCalWaferVOffset = 15;
  static const int32_t kHGCalWaferVMask = 0xF;
  static const int32_t kHGCalWaferVSignOffset = 19;
  static const int32_t kHGCalWaferVSignMask = 0x1;
  static const int32_t kHGCalLayerOffset = 20;
  static const int32_t kHGCalLayerMask = 0x1F;
  static const int32_t kHGCalZsideOffset = 25;
  static const int32_t kHGCalZsideMask = 0x1;
  static const int32_t kHGCalTypeOffset = 26;
  static const int32_t kHGCalTypeMask = 0x3;
};

class HeterogeneousHGCScintillatorDetId {
 public:
  constexpr HeterogeneousHGCScintillatorDetId(uint32_t id): id_(id) {}
  constexpr int type() { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }
  constexpr int zside() const { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }
  constexpr int layer() const { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }

 private:
  uint32_t id_;
  uint32_t kHGCalPhiOffset = 0;
  uint32_t kHGCalPhiMask = 0x1FF;
  uint32_t kHGCalRadiusOffset = 9;
  uint32_t kHGCalRadiusMask = 0xFF;
  uint32_t kHGCalLayerOffset = 17;
  uint32_t kHGCalLayerMask = 0x1F;
  uint32_t kHGCalTriggerOffset = 22;
  uint32_t kHGCalTriggerMask = 0x1;
  uint32_t kHGCalZsideOffset = 25;
  uint32_t kHGCalZsideMask = 0x1;
  uint32_t kHGCalTypeOffset = 26;
  uint32_t kHGCalTypeMask = 0x3;
};

namespace hgcal_conditions {
  namespace parameters {
    enum class HeterogeneousHGCalEEParametersType {Double, Int32_t};
    enum class HeterogeneousHGCalHEFParametersType {Double, Int32_t};
    enum class HeterogeneousHGCalHEBParametersType {Double, Int32_t};

    const std::vector<HeterogeneousHGCalEEParametersType> typesEE = { HeterogeneousHGCalEEParametersType::Double,
								      HeterogeneousHGCalEEParametersType::Double,
								      HeterogeneousHGCalEEParametersType::Double,
								      HeterogeneousHGCalEEParametersType::Double,
								      HeterogeneousHGCalEEParametersType::Int32_t };

    const std::vector<HeterogeneousHGCalHEFParametersType> typesHEF = { HeterogeneousHGCalHEFParametersType::Double,
									HeterogeneousHGCalHEFParametersType::Double,
									HeterogeneousHGCalHEFParametersType::Double,
									HeterogeneousHGCalHEFParametersType::Double,
									HeterogeneousHGCalHEFParametersType::Int32_t };

    const std::vector<HeterogeneousHGCalHEBParametersType> typesHEB = { HeterogeneousHGCalHEBParametersType::Double,
									HeterogeneousHGCalHEBParametersType::Int32_t };

    class HeterogeneousHGCalEEParameters {
    public:
      //indexed by cell number
      double *cellFineX_;
      double *cellFineY_;
      double *cellCoarseX_;
      double *cellCoarseY_;
      //index by wafer number
      int32_t *waferTypeL_;
    };
    class HeterogeneousHGCalHEFParameters {
    public:
      //indexed by cell number
      double *cellFineX_;
      double *cellFineY_;
      double *cellCoarseX_;
      double *cellCoarseY_;
      //index by wafer number
      int32_t *waferTypeL_;
    };
    class HeterogeneousHGCalHEBParameters {
    public:
      double *testD_;
      int32_t *testI_;
    };

  } //namespace parameters

  namespace positions {

    enum class HeterogeneousHGCalPositionsType {Float, Int32_t, Uint32_t};
    
    const std::vector<HeterogeneousHGCalPositionsType> types = { HeterogeneousHGCalPositionsType::Float,
								 HeterogeneousHGCalPositionsType::Float,
								 HeterogeneousHGCalPositionsType::Float,
								 HeterogeneousHGCalPositionsType::Int32_t,
								 HeterogeneousHGCalPositionsType::Uint32_t };
    
    struct HGCalPositionsMapping {
      std::vector<float> z_per_layer;
      std::vector<int32_t> numberCellsHexagon;
      std::vector<uint32_t> detid;
      //variables required for calculating the positions (x,y) from the detid in the GPU
      float waferSize;
      float sensorSeparation;
      //variables required for the mapping of detid -> cell in the geometry
      int32_t firstLayer;
      int32_t lastLayer;
      int32_t waferMax;
      int32_t waferMin;
    };

    struct HeterogeneousHGCalPositionsMapping {
      //the x, y and z positions will not be filled in the CPU
      float* x;
      float* y;
      float* z_per_layer;
      int32_t *numberCellsHexagon;
      uint32_t *detid;
      //variables required for calculating the positions (x,y) from the detid in the GPU
      float waferSize;
      float sensorSeparation;
      //variables required for the mapping of detid -> cell in the geometry
      int32_t firstLayer;
      int32_t lastLayer;
      int32_t waferMax;
      int32_t waferMin;
    };
    
  } //namespace positions
  
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
    size_t nelems_posmap;
  };

} //namespace conditions

#endif //CUDADataFormats_HGCal_HGCConditions_h
