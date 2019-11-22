#ifndef DataFormats_ForwardDetId_HFNoseTriggerDetId_H
#define DataFormats_ForwardDetId_HFNoseTriggerDetId_H 1

#include <iosfwd>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"

/* \brief description of the bit assigment
   [0:3]   u-coordinate of the cell (measured from the lower left
   [4:7]   v-coordinate of the cell  corner of the wafer)
   [8:11]  abs(u) of the wafer (u-axis points along -x axis)
   [12:12] sign of u (0:+u; 1:-u) (u=0 is at the center of beam line)
   [13:16] abs(v) of the wafer (v-axis points 60-degree wrt x-axis)
   [17:17] sign of v (0:+v; 1:-v) (v=0 is at the center of beam line)
   [18:22] layer number 
   [23:24] Type (0 fine divisions of wafer with 120 mum thick silicon
                 1 coarse divisions of wafer with 200 mum thick silicon
                 2 coarse divisions of wafer with 300 mum thick silicon)
   [25:26] Subdetector Type (HFNoseTrigger)
   [27:27] z-side (0 for +z; 1 for -z)
   [28:31] Detector type (HGCalTrigger)
*/

class HFNoseTriggerDetId : public DetId {
public:
  static const int HFNoseTriggerCell = 4;

  /** Create a null cellid*/
  HFNoseTriggerDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HFNoseTriggerDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HFNoseTriggerDetId(int subdet, int zp, int type, int layer, int waferU, int waferV, int cellU, int cellV);
  /** Constructor from a generic cell id */
  HFNoseTriggerDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HFNoseTriggerDetId& operator=(const DetId& id);

  /// get the subdetector
  HGCalTriggerSubdetector subdet() const {
    return (HGCalTriggerSubdetector)((id_ >> kHFNoseSubdetOffset) & kHFNoseSubdetMask);
  }

  /// get the type
  int type() const { return (id_ >> kHFNoseTypeOffset) & kHFNoseTypeMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return (((id_ >> kHFNoseZsideOffset) & kHFNoseZsideMask) ? -1 : 1); }

  /// get the layer #
  int layer() const { return (id_ >> kHFNoseLayerOffset) & kHFNoseLayerMask; }

  /** Converter for a geometry cell id */
  HFNoseDetId geometryCell() const { return HFNoseDetId(zside(), 0, layer(), waferU(), waferV(), 0, 0); }
  HFNoseDetId moduleId() const { return HFNoseDetId(zside(), type(), layer(), waferU(), waferV(), 0, 0); }

  /// get the cell #'s in u,v or in x,y
  int triggerCellU() const { return (id_ >> kHFNoseCellUOffset) & kHFNoseCellUMask; }
  int triggerCellV() const { return (id_ >> kHFNoseCellVOffset) & kHFNoseCellVMask; }
  std::pair<int, int> triggerCellUV() const { return std::pair<int, int>(triggerCellU(), triggerCellV()); }
  int triggerCellX() const;
  int triggerCellY() const;
  std::pair<int, int> triggerCellXY() const { return std::pair<int, int>(triggerCellX(), triggerCellY()); }

  /// get the wafer #'s in u,v or in x,y
  int waferUAbs() const { return (id_ >> kHFNoseWaferUOffset) & kHFNoseWaferUMask; }
  int waferVAbs() const { return (id_ >> kHFNoseWaferVOffset) & kHFNoseWaferVMask; }
  int waferU() const {
    return (((id_ >> kHFNoseWaferUSignOffset) & kHFNoseWaferUSignMask) ? -waferUAbs() : waferUAbs());
  }
  int waferV() const {
    return (((id_ >> kHFNoseWaferVSignOffset) & kHFNoseWaferVSignMask) ? -waferVAbs() : waferVAbs());
  }
  std::pair<int, int> waferUV() const { return std::pair<int, int>(waferU(), waferV()); }
  int waferX() const { return (-2 * waferU() + waferV()); }
  int waferY() const { return (2 * waferV()); }
  std::pair<int, int> waferXY() const { return std::pair<int, int>(waferX(), waferY()); }

  // get trigger cell u,v
  std::vector<int> cellU() const;
  std::vector<int> cellV() const;
  std::vector<std::pair<int, int> > cellUV() const;

  /// consistency check : no bits left => no overhead
  bool isEE() const { return (layer() <= kHFNoseMaxEELayer); }
  bool isHSilicon() const { return (layer() > kHFNoseMaxEELayer); }
  bool isForward() const { return true; }

  static const HFNoseTriggerDetId Undefined;

  static const int kHFNoseCellUOffset = 0;
  static const int kHFNoseCellUMask = 0xF;
  static const int kHFNoseCellVOffset = 4;
  static const int kHFNoseCellVMask = 0xF;
  static const int kHFNoseWaferUOffset = 8;
  static const int kHFNoseWaferUMask = 0xF;
  static const int kHFNoseWaferUSignOffset = 12;
  static const int kHFNoseWaferUSignMask = 0x1;
  static const int kHFNoseWaferVOffset = 13;
  static const int kHFNoseWaferVMask = 0xF;
  static const int kHFNoseWaferVSignOffset = 17;
  static const int kHFNoseWaferVSignMask = 0x1;
  static const int kHFNoseLayerOffset = 18;
  static const int kHFNoseLayerMask = 0x1F;
  static const int kHFNoseTypeOffset = 23;
  static const int kHFNoseTypeMask = 0x3;
  static const int kHFNoseZsideOffset = 27;
  static const int kHFNoseZsideMask = 0x1;
  static const int kHFNoseSubdetOffset = 25;
  static const int kHFNoseSubdetMask = 0x3;
  static const int kHFNoseMaxEELayer = 6;
};

std::ostream& operator<<(std::ostream&, const HFNoseTriggerDetId& id);

#endif
