#ifndef CUDADataFormats_HGCal_HGCConditions_h
#define CUDADataFormats_HGCal_HGCConditions_h

class HeterogeneousHGCSiliconDetId {
 public:
  constexpr HeterogeneousHGCSiliconDetId(uint32_t id): id_(id) {}
  constexpr uint32_t type() { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }
  constexpr uint32_t zside() { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }
  constexpr uint32_t layer() { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }
  constexpr uint32_t waferUAbs() { return (id_ >> kHGCalWaferUOffset) & kHGCalWaferUMask; }
  constexpr uint32_t waferVAbs() { return (id_ >> kHGCalWaferVOffset) & kHGCalWaferVMask; }
  constexpr uint32_t waferU() { return (((id_ >> kHGCalWaferUSignOffset) & kHGCalWaferUSignMask) ? -waferUAbs() : waferUAbs()); }
  constexpr uint32_t waferV() { return (((id_ >> kHGCalWaferVSignOffset) & kHGCalWaferVSignMask) ? -waferVAbs() : waferVAbs()); }
  constexpr uint32_t cellU() { return (id_ >> kHGCalCellUOffset) & kHGCalCellUMask; }
  constexpr uint32_t cellV() { return (id_ >> kHGCalCellVOffset) & kHGCalCellVMask; }
  constexpr int cellX() { 
    int N = (type() == HGCalFine) ? HGCalFineN : HGCalCoarseN;
    return (3 * (cellV() - N) + 2);
  }
  constexpr int cellY() {
    int N = (type() == HGCalFine) ? HGCalFineN : HGCalCoarseN;
    return (2 * cellU() - (N + cellV()));
  }
  
 private:
  uint32_t id_;
  enum waferType { HGCalFine = 0, HGCalCoarseThin = 1, HGCalCoarseThick = 2 };
  int HGCalFineN = 12;
  int HGCalCoarseN = 8;
  int kHGCalCellUOffset = 0;
  int kHGCalCellUMask = 0x1F;
  int kHGCalCellVOffset = 5;
  int kHGCalCellVMask = 0x1F;
  int kHGCalWaferUOffset = 10;
  int kHGCalWaferUMask = 0xF;
  int kHGCalWaferUSignOffset = 14;
  int kHGCalWaferUSignMask = 0x1;
  int kHGCalWaferVOffset = 15;
  int kHGCalWaferVMask = 0xF;
  int kHGCalWaferVSignOffset = 19;
  int kHGCalWaferVSignMask = 0x1;
  int kHGCalLayerOffset = 20;
  int kHGCalLayerMask = 0x1F;
  int kHGCalZsideOffset = 25;
  int kHGCalZsideMask = 0x1;
  int kHGCalTypeOffset = 26;
  int kHGCalTypeMask = 0x3;
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

  struct HeterogeneousEEConditionsESProduct {
    parameters::HeterogeneousHGCalEEParameters params;
  };
  struct HeterogeneousHEFConditionsESProduct {
    parameters::HeterogeneousHGCalHEFParameters params;
  };
  struct HeterogeneousHEBConditionsESProduct {
    parameters::HeterogeneousHGCalHEBParameters params;
  };

} //namespace conditions

#endif
