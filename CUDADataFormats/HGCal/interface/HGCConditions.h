#ifndef CUDADataFormats_HGCal_HGCConditions_h
#define CUDADataFormats_HGCal_HGCConditions_h

class HeterogeneousHGCalDetId {
 public:
  constexpr HeterogeneousHGCalDetId(uint32_t id): id_(id) {}
  constexpr uint32_t cell()      { return id_ & kHGCalCellMask; }
  constexpr uint32_t wafer()     { return (id_ >> kHGCalWaferOffset) & kHGCalWaferMask; }
  constexpr uint32_t layer()     { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }
  //CAREFUL: is this wafertype the right one? In the code waferTypeL_ from HGCalParameters is used
  constexpr uint32_t waferType() { return ((id_ >> kHGCalWaferTypeOffset) & kHGCalWaferTypeMask ? 1 : -1); }

 private:
  uint32_t id_;
  uint32_t kHGCalCellOffset = 0;
  uint32_t kHGCalCellMask = 0xFF;
  uint32_t kHGCalWaferOffset = 8;
  uint32_t kHGCalWaferMask = 0x3FF;
  uint32_t kHGCalWaferTypeOffset = 18;
  uint32_t kHGCalWaferTypeMask = 0x1;
  uint32_t kHGCalLayerOffset = 19;
  uint32_t kHGCalLayerMask = 0x1F;
  uint32_t kHGCalZsideOffset = 24;
  uint32_t kHGCalZsideMask = 0x1;
  uint32_t kHGCalMaskCell = 0xFFFBFF00;  
};

namespace hgcal_conditions {
  namespace parameters {
    enum class HeterogeneousHGCalParametersType {Double, Int32_t};
    const std::vector<HeterogeneousHGCalParametersType> types = { HeterogeneousHGCalParametersType::Double,
								  HeterogeneousHGCalParametersType::Double,
								  HeterogeneousHGCalParametersType::Double,
								  HeterogeneousHGCalParametersType::Double,
								  HeterogeneousHGCalParametersType::Int32_t };
    class HeterogeneousHGCalParameters {
    public:
      //indexed by cell number
      double *cellFineX_;
      double *cellFineY_;
      double *cellCoarseX_;
      double *cellCoarseY_;
      //index by wafer number
      int32_t *waferTypeL_;
    };
  } //namespace parameters

  struct HeterogeneousHEFConditionsESProduct {
    //memory block #1
    parameters::HeterogeneousHGCalParameters params;
  };

} //namespace conditions

#endif
