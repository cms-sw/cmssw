#ifndef DataFormats_ForwardDetId_HGCSiliconDetId_H
#define DataFormats_ForwardDetId_HGCSiliconDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

/* \brief description of the bit assigment
   [0:4]   u-coordinate of the cell (measured from the lower left
   [5:9]   v-coordinate of the cell  corner of the wafer)
   [10:13] abs(u) of the wafer (u-axis points along -x axis)
   [14:14] sign of u (0:+u; 1:-u) (u=0 is at the center of beam line)
   [15:18] abs(v) of the wafer (v-axis points 60-degree wrt x-axis)
   [19:19] sign of v (0:+v; 1:-v) (v=0 is at the center of beam line)
   [20:24] layer number 
   [25:25] z-side (0 for +z; 1 for -z)
   [26:27] Type (0 fine divisions of wafer with 120 mum thick silicon
                 1 coarse divisions of wafer with 200 mum thick silicon
                 2 coarse divisions of wafer with 300 mum thick silicon
                 3 fine divisions of wafer with 200 mum thick silicon)
   [28:31] Detector type (HGCalEE or HGCalHSi)
*/
class HGCSiliconDetId : public DetId {
public:
  enum waferType { HGCalFine = 0, HGCalCoarseThin = 1, HGCalCoarseThick = 2, HGCalFineThick = 3 };
  static constexpr int32_t HGCalHighDensityN = 12;
  static constexpr int32_t HGCalLowDensityN = 8;
  static constexpr int32_t HGCalFineTrigger = 3;
  static constexpr int32_t HGCalCoarseTrigger = 2;
  static constexpr int32_t HGCal0Depletion = 120;
  static constexpr int32_t HGCal1Depletion = 200;
  static constexpr int32_t HGCal2Depletion = 300;

  /** Create a null cellid*/
  constexpr HGCSiliconDetId() : DetId() {}
  /** Create cellid from raw id (0=invalid tower id) */
  constexpr HGCSiliconDetId(uint32_t rawid) : DetId(rawid) {}
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  constexpr HGCSiliconDetId(DetId::Detector det,
                            int32_t zp,
                            int32_t type,
                            int32_t layer,
                            int32_t waferU,
                            int32_t waferV,
                            int32_t cellU,
                            int32_t cellV)
      : DetId(det, ForwardEmpty) {
    int32_t waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
    int32_t waferUsign = (waferU >= 0) ? 0 : 1;
    int32_t waferVsign = (waferV >= 0) ? 0 : 1;
    int32_t zside = (zp < 0) ? 1 : 0;
    id_ |= (((cellU & kHGCalCellUMask) << kHGCalCellUOffset) | ((cellV & kHGCalCellVMask) << kHGCalCellVOffset) |
            ((waferUabs & kHGCalWaferUMask) << kHGCalWaferUOffset) |
            ((waferUsign & kHGCalWaferUSignMask) << kHGCalWaferUSignOffset) |
            ((waferVabs & kHGCalWaferVMask) << kHGCalWaferVOffset) |
            ((waferVsign & kHGCalWaferVSignMask) << kHGCalWaferVSignOffset) |
            ((layer & kHGCalLayerMask) << kHGCalLayerOffset) | ((zside & kHGCalZsideMask) << kHGCalZsideOffset) |
            ((type & kHGCalTypeMask) << kHGCalTypeOffset));
  }

  /** Constructor from a generic cell id */
  constexpr HGCSiliconDetId(const DetId& gen) {
    if (!gen.null()) {
      if ((gen.det() != HGCalEE) && (gen.det() != HGCalHSi)) {
        throw cms::Exception("Invalid DetId")
            << "Cannot initialize HGCSiliconDetId from " << std::hex << gen.rawId() << std::dec;
      }
    }
    id_ = gen.rawId();
  }

  /** Assignment from a generic cell id */
  constexpr HGCSiliconDetId& operator=(const DetId& gen) {
    if (!gen.null()) {
      if ((gen.det() != HGCalEE) && (gen.det() != HGCalHSi)) {
        throw cms::Exception("Invalid DetId")
            << "Cannot assign HGCSiliconDetId from " << std::hex << gen.rawId() << std::dec;
      }
    }
    id_ = gen.rawId();
    return (*this);
  }

  /** Converter for a geometry cell id */
  constexpr HGCSiliconDetId geometryCell() const {
    return HGCSiliconDetId(det(), zside(), 0, layer(), waferU(), waferV(), 0, 0);
  }
  constexpr HGCSiliconDetId moduleId() const {
    return HGCSiliconDetId(det(), zside(), type(), layer(), waferU(), waferV(), 0, 0);
  }

  /// get the subdetector
  constexpr DetId::Detector subdet() const { return det(); }

  /// get the type
  constexpr int32_t type() const { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }
  constexpr bool lowDensity() const { return ((type() == HGCalCoarseThin) || (type() == HGCalCoarseThick)); }
  constexpr bool highDensity() const { return ((type() == HGCalFine) || (type() == HGCalFineThick)); }
  constexpr int32_t depletion() const {
    return ((type() == HGCalFine) ? HGCal0Depletion
                                  : ((type() == HGCalCoarseThick) ? HGCal2Depletion : HGCal1Depletion));
  }

  /// get the z-side of the cell (1/-1)
  constexpr int32_t zside() const { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }

  /// get the layer #
  constexpr int32_t layer() const { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }

  /// get the cell #'s in u,v or in x,y
  constexpr int32_t cellU() const { return (id_ >> kHGCalCellUOffset) & kHGCalCellUMask; }
  constexpr int32_t cellV() const { return (id_ >> kHGCalCellVOffset) & kHGCalCellVMask; }
  constexpr std::pair<int32_t, int32_t> cellUV() const { return std::pair<int32_t, int32_t>(cellU(), cellV()); }
  constexpr int32_t cellX() const {
    int32_t N = ((type() == HGCalFine) || (type() == HGCalFineThick)) ? HGCalHighDensityN : HGCalLowDensityN;
    return (3 * (cellV() - N) + 2);
  }
  constexpr int32_t cellY() const {
    int32_t N = ((type() == HGCalFine) || (type() == HGCalFineThick)) ? HGCalHighDensityN : HGCalLowDensityN;
    return (2 * cellU() - (N + cellV()));
  }
  constexpr std::pair<int32_t, int32_t> cellXY() const { return std::pair<int32_t, int32_t>(cellX(), cellY()); }

  /// get the wafer #'s in u,v or in x,y
  constexpr int32_t waferUAbs() const { return (id_ >> kHGCalWaferUOffset) & kHGCalWaferUMask; }
  constexpr int32_t waferVAbs() const { return (id_ >> kHGCalWaferVOffset) & kHGCalWaferVMask; }
  constexpr int32_t waferU() const {
    return (((id_ >> kHGCalWaferUSignOffset) & kHGCalWaferUSignMask) ? -waferUAbs() : waferUAbs());
  }
  constexpr int32_t waferV() const {
    return (((id_ >> kHGCalWaferVSignOffset) & kHGCalWaferVSignMask) ? -waferVAbs() : waferVAbs());
  }
  constexpr std::pair<int32_t, int32_t> waferUV() const { return std::pair<int32_t, int32_t>(waferU(), waferV()); }
  constexpr int32_t waferX() const { return (-2 * waferU() + waferV()); }
  constexpr int32_t waferY() const { return (2 * waferV()); }
  constexpr std::pair<int32_t, int32_t> waferXY() const { return std::pair<int32_t, int32_t>(waferX(), waferY()); }
  constexpr void unpack(
      int32_t& ty, int32_t& zs, int32_t& ly, int32_t& wU, int32_t& wV, int32_t& cU, int32_t& cV) const {
    ty = type();
    zs = zside();
    ly = layer();
    wU = waferU();
    wV = waferV();
    cU = cellU();
    cV = cellV();
  }

  // get trigger cell u,v
  constexpr int32_t triggerCellU() const {
    int32_t N = ((type() == HGCalFine) || (type() == HGCalFineThick)) ? HGCalHighDensityN : HGCalLowDensityN;
    int32_t NT = ((type() == HGCalFine) || (type() == HGCalFineThick)) ? HGCalFineTrigger : HGCalCoarseTrigger;
    return (cellU() >= N && cellV() >= N)
               ? cellU() / NT
               : ((cellU() < N && cellU() <= cellV()) ? cellU() / NT : (1 + (cellU() - (cellV() % NT + 1)) / NT));
  }
  constexpr int32_t triggerCellV() const {
    int32_t N = ((type() == HGCalFine) || (type() == HGCalFineThick)) ? HGCalHighDensityN : HGCalLowDensityN;
    int32_t NT = ((type() == HGCalFine) || (type() == HGCalFineThick)) ? HGCalFineTrigger : HGCalCoarseTrigger;
    return (cellU() >= N && cellV() >= N)
               ? cellV() / NT
               : ((cellU() < N && cellU() <= cellV()) ? ((cellV() - cellU()) / NT + cellU() / NT) : cellV() / NT);
  }
  constexpr std::pair<int32_t, int32_t> triggerCellUV() const {
    return std::pair<int32_t, int32_t>(triggerCellU(), triggerCellV());
  }

  /// consistency check : no bits left => no overhead
  constexpr bool isEE() const { return (det() == HGCalEE); }
  constexpr bool isHE() const { return (det() == HGCalHSi); }
  constexpr bool isForward() const { return true; }

  static const HGCSiliconDetId Undefined;

public:
  static constexpr uint32_t kHGCalCellUOffset = 0;
  static constexpr uint32_t kHGCalCellUMask = 0x1F;
  static constexpr uint32_t kHGCalCellVOffset = 5;
  static constexpr uint32_t kHGCalCellVMask = 0x1F;
  static constexpr uint32_t kHGCalWaferUOffset = 10;
  static constexpr uint32_t kHGCalWaferUMask = 0xF;
  static constexpr uint32_t kHGCalWaferUSignOffset = 14;
  static constexpr uint32_t kHGCalWaferUSignMask = 0x1;
  static constexpr uint32_t kHGCalWaferVOffset = 15;
  static constexpr uint32_t kHGCalWaferVMask = 0xF;
  static constexpr uint32_t kHGCalWaferVSignOffset = 19;
  static constexpr uint32_t kHGCalWaferVSignMask = 0x1;
  static constexpr uint32_t kHGCalLayerOffset = 20;
  static constexpr uint32_t kHGCalLayerMask = 0x1F;
  static constexpr uint32_t kHGCalZsideOffset = 25;
  static constexpr uint32_t kHGCalZsideMask = 0x1;
  static constexpr uint32_t kHGCalTypeOffset = 26;
  static constexpr uint32_t kHGCalTypeMask = 0x3;
};

std::ostream& operator<<(std::ostream&, const HGCSiliconDetId& id);

#endif
