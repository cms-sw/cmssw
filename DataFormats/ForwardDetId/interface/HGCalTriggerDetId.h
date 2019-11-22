#ifndef DataFormats_ForwardDetId_HGCalTriggerDetId_H
#define DataFormats_ForwardDetId_HGCalTriggerDetId_H 1

#include <iosfwd>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

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
   [25:26] Subdetector Type (HGCalEETrigger/HGCalHSiTrigger)
   [27:27] z-side (0 for +z; 1 for -z)
   [28:31] Detector type (HGCalTrigger)
*/

class HGCalTriggerDetId : public DetId {
public:
  static const int HGCalTriggerCell = 4;

  /** Create a null cellid*/
  HGCalTriggerDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCalTriggerDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCalTriggerDetId(int subdet, int zp, int type, int layer, int waferU, int waferV, int cellU, int cellV);
  /** Constructor from a generic cell id */
  HGCalTriggerDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCalTriggerDetId& operator=(const DetId& id);

  /// get the subdetector
  HGCalTriggerSubdetector subdet() const {
    return (HGCalTriggerSubdetector)((id_ >> kHGCalSubdetOffset) & kHGCalSubdetMask);
  }

  /// get the type
  int type() const { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }

  /// get the layer #
  int layer() const { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }

  /** Converter for a geometry cell id */
  HGCSiliconDetId geometryCell() const { return HGCSiliconDetId(det(), zside(), 0, layer(), waferU(), waferV(), 0, 0); }
  HGCSiliconDetId moduleId() const {
    return HGCSiliconDetId(det(), zside(), type(), layer(), waferU(), waferV(), 0, 0);
  }

  /// get the cell #'s in u,v or in x,y
  int triggerCellU() const { return (id_ >> kHGCalCellUOffset) & kHGCalCellUMask; }
  int triggerCellV() const { return (id_ >> kHGCalCellVOffset) & kHGCalCellVMask; }
  std::pair<int, int> triggerCellUV() const { return std::pair<int, int>(triggerCellU(), triggerCellV()); }
  int triggerCellX() const;
  int triggerCellY() const;
  std::pair<int, int> triggerCellXY() const { return std::pair<int, int>(triggerCellX(), triggerCellY()); }

  /// get the wafer #'s in u,v or in x,y
  int waferUAbs() const { return (id_ >> kHGCalWaferUOffset) & kHGCalWaferUMask; }
  int waferVAbs() const { return (id_ >> kHGCalWaferVOffset) & kHGCalWaferVMask; }
  int waferU() const { return (((id_ >> kHGCalWaferUSignOffset) & kHGCalWaferUSignMask) ? -waferUAbs() : waferUAbs()); }
  int waferV() const { return (((id_ >> kHGCalWaferVSignOffset) & kHGCalWaferVSignMask) ? -waferVAbs() : waferVAbs()); }
  std::pair<int, int> waferUV() const { return std::pair<int, int>(waferU(), waferV()); }
  int waferX() const { return (-2 * waferU() + waferV()); }
  int waferY() const { return (2 * waferV()); }
  std::pair<int, int> waferXY() const { return std::pair<int, int>(waferX(), waferY()); }

  // get trigger cell u,v
  std::vector<int> cellU() const;
  std::vector<int> cellV() const;
  std::vector<std::pair<int, int> > cellUV() const;

  /// consistency check : no bits left => no overhead
  bool isEE() const { return (subdet() == HGCalEETrigger); }
  bool isHSilicon() const { return (subdet() == HGCalHSiTrigger); }
  bool isForward() const { return true; }

  static const HGCalTriggerDetId Undefined;

  static const int kHGCalCellUOffset = 0;
  static const int kHGCalCellUMask = 0xF;
  static const int kHGCalCellVOffset = 4;
  static const int kHGCalCellVMask = 0xF;
  static const int kHGCalWaferUOffset = 8;
  static const int kHGCalWaferUMask = 0xF;
  static const int kHGCalWaferUSignOffset = 12;
  static const int kHGCalWaferUSignMask = 0x1;
  static const int kHGCalWaferVOffset = 13;
  static const int kHGCalWaferVMask = 0xF;
  static const int kHGCalWaferVSignOffset = 17;
  static const int kHGCalWaferVSignMask = 0x1;
  static const int kHGCalLayerOffset = 18;
  static const int kHGCalLayerMask = 0x1F;
  static const int kHGCalTypeOffset = 23;
  static const int kHGCalTypeMask = 0x3;
  static const int kHGCalZsideOffset = 27;
  static const int kHGCalZsideMask = 0x1;
  static const int kHGCalSubdetOffset = 25;
  static const int kHGCalSubdetMask = 0x3;
};

std::ostream& operator<<(std::ostream&, const HGCalTriggerDetId& id);

#endif
