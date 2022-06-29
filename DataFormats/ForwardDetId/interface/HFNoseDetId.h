#ifndef DataFormats_ForwardDetId_HFNoseDetId_H
#define DataFormats_ForwardDetId_HFNoseDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

/* \brief description of the bit assigment
   [0:4]   u-coordinate of the cell (measured from the lower left
   [5:9]   v-coordinate of the cell  corner of the wafer)
   [10:13] abs(u) of the wafer (u-axis points along -x axis)
   [14:14] sign of u (0:+u; 1:-u) (u=0 is at the center of beam line)
   [15:18] abs(v) of the wafer (v-axis points 60-degree wrt x-axis)
   [19:19] sign of v (0:+v; 1:-v) (v=0 is at the center of beam line)
   [20:22] layer number - 1
   [23:23] z-side (0 for +z; 1 for -z)
   [24:24] Type (0 fine divisions of wafer with 120 mum thick silicon
                 1 coarse divisions of wafer with 200 mum thick silicon)
   [25:27] Subdetector type (HFNose)
   [28:31] Detector type (Forward)
*/
class HFNoseDetId : public DetId {
public:
  enum hfNoseWaferType { HFNoseFine = 0, HFNoseCoarseThin = 1, HFNoseCoarseThick = 2 };
  static const int HFNoseFineN = 12;
  static const int HFNoseCoarseN = 8;
  static const int HFNoseFineTrigger = 3;
  static const int HFNoseCoarseTrigger = 2;
  static const int HFNoseLayerEEmax = 6;

  /** Create a null cellid*/
  HFNoseDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HFNoseDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HFNoseDetId(DetId::Detector det, int zp, int type, int layer, int waferU, int waferV, int cellU, int cellV);
  HFNoseDetId(int zp, int type, int layer, int waferU, int waferV, int cellU, int cellV);
  /** Constructor from a generic cell id */
  HFNoseDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HFNoseDetId& operator=(const DetId& id);

  /// get the subdetector
  ForwardSubdetector subdet() const { return HFNose; }

  /** Converter for a geometry cell id */
  HFNoseDetId geometryCell() const { return HFNoseDetId(zside(), 0, layer(), waferU(), waferV(), 0, 0); }
  HFNoseDetId moduleId() const { return HFNoseDetId(zside(), type(), layer(), waferU(), waferV(), 0, 0); }

  /// get the type
  int type() const { return (id_ >> kHFNoseTypeOffset) & kHFNoseTypeMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return (((id_ >> kHFNoseZsideOffset) & kHFNoseZsideMask) ? -1 : 1); }

  /// get the layer #
  int layer() const { return ((id_ >> kHFNoseLayerOffset) & kHFNoseLayerMask) + 1; }

  /// get the cell #'s in u,v or in x,y
  int cellU() const { return (id_ >> kHFNoseCellUOffset) & kHFNoseCellUMask; }
  int cellV() const { return (id_ >> kHFNoseCellVOffset) & kHFNoseCellVMask; }
  std::pair<int, int> cellUV() const { return std::pair<int, int>(cellU(), cellV()); }
  int cellX() const {
    int N = (type() == HFNoseFine) ? HFNoseFineN : HFNoseCoarseN;
    return (3 * (cellV() - N) + 2);
  }
  int cellY() const {
    int N = (type() == HFNoseFine) ? HFNoseFineN : HFNoseCoarseN;
    return (2 * cellU() - (N + cellV()));
  }
  std::pair<int, int> cellXY() const { return std::pair<int, int>(cellX(), cellY()); }

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
  int triggerCellU() const {
    int N = (type() == HFNoseFine) ? HFNoseFineN : HFNoseCoarseN;
    int NT = (type() == HFNoseFine) ? HFNoseFineTrigger : HFNoseCoarseTrigger;
    return (cellU() >= N && cellV() >= N)
               ? cellU() / NT
               : ((cellU() < N && cellU() <= cellV()) ? cellU() / NT : (1 + (cellU() - (cellV() % NT + 1)) / NT));
  }
  int triggerCellV() const {
    int N = (type() == HFNoseFine) ? HFNoseFineN : HFNoseCoarseN;
    int NT = (type() == HFNoseFine) ? HFNoseFineTrigger : HFNoseCoarseTrigger;
    return (cellU() >= N && cellV() >= N)
               ? cellV() / NT
               : ((cellU() < N && cellU() <= cellV()) ? ((cellV() - cellU()) / NT + cellU() / NT) : cellV() / NT);
  }
  std::pair<int, int> triggerCellUV() const { return std::pair<int, int>(triggerCellU(), triggerCellV()); }

  /// consistency check : no bits left => no overhead
  bool isEE() const { return (layer() <= HFNoseLayerEEmax); }
  bool isHE() const { return (layer() > HFNoseLayerEEmax); }
  bool isForward() const { return true; }

  static const HFNoseDetId Undefined;

private:
  static const int kHFNoseCellUOffset = 0;
  static const int kHFNoseCellUMask = 0x1F;
  static const int kHFNoseCellVOffset = 5;
  static const int kHFNoseCellVMask = 0x1F;
  static const int kHFNoseWaferUOffset = 10;
  static const int kHFNoseWaferUMask = 0xF;
  static const int kHFNoseWaferUSignOffset = 14;
  static const int kHFNoseWaferUSignMask = 0x1;
  static const int kHFNoseWaferVOffset = 15;
  static const int kHFNoseWaferVMask = 0xF;
  static const int kHFNoseWaferVSignOffset = 19;
  static const int kHFNoseWaferVSignMask = 0x1;
  static const int kHFNoseLayerOffset = 20;
  static const int kHFNoseLayerMask = 0x7;
  static const int kHFNoseZsideOffset = 23;
  static const int kHFNoseZsideMask = 0x1;
  static const int kHFNoseTypeOffset = 24;
  static const int kHFNoseTypeMask = 0x1;
};

std::ostream& operator<<(std::ostream&, const HFNoseDetId& id);

#endif
