#ifndef DataFormats_ForwardDetId_HGCScintillatorDetId_H
#define DataFormats_ForwardDetId_HGCScintillatorDetId_H 1

#include <iosfwd>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

/* \brief description of the bit assigment
   [0:8]   iphi index wrt x-axis on +z side
   [9:16]  |radius| index (starting from a minimum radius depending on type)
   [17:21] Layer #
   [22]    Trigger(1)/Detector(0) cell
   [23]    SiPM type (0 for 2mm: 1 for 4mm)
   [24]    Free
   [25:25] z-side (0 for +z; 1 for -z)
   [26:27] Tile granularity and type (0 fine divisions of scintillators;
                                      1 coarse divisions of type "c";
                                      2 coarse divisions of type "m")
   [28:31] Detector type (HGCalHSc)
*/

class HGCScintillatorDetId : public DetId {
public:
  /** Create a null cellid*/
  HGCScintillatorDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCScintillatorDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCScintillatorDetId(int type, int layer, int iradius, int iphi, bool trigger = false, int sipm = 0);
  /** Constructor from a generic cell id */
  HGCScintillatorDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCScintillatorDetId& operator=(const DetId& id);

  /** Converter for a geometry cell id */
  HGCScintillatorDetId geometryCell() const;

  /// get the subdetector
  DetId::Detector subdet() const { return det(); }

  /// get/set the type
  int type() const { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }
  void setType(int type);

  /// get the z-side of the cell (1/-1)
  int zside() const { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }

  /// get the layer #
  int layer() const { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }

  /// get the eta index
  int iradiusAbs() const;
  int iradius() const { return zside() * iradiusAbs(); }
  int ietaAbs() const { return iradiusAbs(); }
  int ieta() const { return zside() * ietaAbs(); }

  /// get the phi index
  int iphi() const;
  std::pair<int, int> ietaphi() const { return std::pair<int, int>(ieta(), iphi()); }
  std::pair<int, int> iradiusphi() const { return std::pair<int, int>(iradius(), iphi()); }

  /// get/set the sipm size
  int sipm() const { return (id_ >> kHGCalSiPMOffset) & kHGCalSiPMMask; }
  void setSiPM(int sipm);

  /// trigger or detector cell
  std::vector<HGCScintillatorDetId> detectorCells() const;
  bool trigger() const { return (((id_ >> kHGCalTriggerOffset) & kHGCalTriggerMask) == 1); }
  HGCScintillatorDetId triggerCell() const;

  /// consistency check : no bits left => no overhead
  bool isEE() const { return false; }
  bool isHE() const { return true; }
  bool isForward() const { return true; }

  static const HGCScintillatorDetId Undefined;

public:
  static const int kHGCalPhiOffset = 0;
  static const int kHGCalPhiMask = 0x1FF;
  static const int kHGCalRadiusOffset = 9;
  static const int kHGCalRadiusMask = 0xFF;
  static const int kHGCalLayerOffset = 17;
  static const int kHGCalLayerMask = 0x1F;
  static const int kHGCalTriggerOffset = 22;
  static const int kHGCalTriggerMask = 0x1;
  static const int kHGCalSiPMOffset = 23;
  static const int kHGCalSiPMMask = 0x1;
  static const int kHGCalSiPMMask0 = 0xFF7FFFFF;
  static const int kHGCalZsideOffset = 25;
  static const int kHGCalZsideMask = 0x1;
  static const int kHGCalTypeOffset = 26;
  static const int kHGCalTypeMask = 0x3;
  static const int kHGCalTypeMask0 = 0xF3FFFFFF;

  int iradiusTriggerAbs() const;
  int iradiusTrigger() const { return zside() * iradiusTriggerAbs(); }
  int iphiTrigger() const;
};

std::ostream& operator<<(std::ostream&, const HGCScintillatorDetId& id);

#endif
