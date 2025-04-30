#ifndef DataFormats_ForwardDetId_HGCScintillatorDetId_H
#define DataFormats_ForwardDetId_HGCScintillatorDetId_H 1

#include <iosfwd>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

/* \brief description of the bit assigment
   [0:8]   iphi index wrt x-axis on +z side
   [9:16]  |ring| index (starting from a minimum radius depending on type)
   [17:21] Layer #
   [22]    Trigger(1)/Detector(0) cell
   [23]    SiPM type (0 for 2mm or 9mm: 1 for 4mm)
   [24]    Granularity of the tile (0 normal; 1 fine)
   [25:25] z-side (0 for +z; 1 for -z)
   [26:27] Tile make (1 of type "c"; 2 of type "m")
   [28:31] Detector type (HGCalHSc)
*/

class HGCScintillatorDetId : public DetId {
public:
  enum tileGranularity { HGCalTileNormal = 0, HGCalTileFine = 1 };
  enum sipmType { HGCalSiPMSmall = 0, HGCalSiPMLarge = 1 };
  enum tileType { HGCalTileTypeUnknown = 0, HGCalTileTypeCaste = 1, HGCalTileTypeMould = 2 };
  /** Create a null cellid*/
  constexpr HGCScintillatorDetId() : DetId() {}
  /** Create cellid from raw id (0=invalid tower id) */
  constexpr HGCScintillatorDetId(uint32_t rawid) : DetId(rawid) {}
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  constexpr HGCScintillatorDetId(
      int type, int layer, int ring, int phi, bool trigger = false, int sipm = 0, int granularity = 0)
      : DetId(HGCalHSc, ForwardEmpty) {
    int zside = (ring < 0) ? 1 : 0;
    int itrig = trigger ? 1 : 0;
    int ringAbs = std::abs(ring);
    id_ |= (((type & kHGCalTypeMask) << kHGCalTypeOffset) | ((zside & kHGCalZsideMask) << kHGCalZsideOffset) |
            ((sipm & kHGCalSiPMMask) << kHGCalSiPMOffset) | ((itrig & kHGCalTriggerMask) << kHGCalTriggerOffset) |
            ((layer & kHGCalLayerMask) << kHGCalLayerOffset) | ((ringAbs & kHGCalRadiusMask) << kHGCalRadiusOffset) |
            ((phi & kHGCalPhiMask) << kHGCalPhiOffset) |
            ((granularity & kHGCalGranularityMask) << kHGCalGranularityOffset));
  }

  /** Constructor from a generic cell id */
  constexpr HGCScintillatorDetId(const DetId& gen) {
    if (!gen.null()) {
      if (gen.det() != HGCalHSc) {
        throw cms::Exception("Invalid DetId")
            << "Cannot initialize HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec;
      }
    }
    id_ = gen.rawId();
  }

  /** Assignment from a generic cell id */
  constexpr HGCScintillatorDetId& operator=(const DetId& gen) {
    if (!gen.null()) {
      if (gen.det() != HGCalHSc) {
        throw cms::Exception("Invalid DetId")
            << "Cannot assign HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec;
      }
    }
    id_ = gen.rawId();
    return (*this);
  }

  /** Converter for a geometry cell id */
  constexpr HGCScintillatorDetId geometryCell() const {
    if (trigger()) {
      return HGCScintillatorDetId(type(), layer(), iradiusTrigger(), iphiTrigger(), false);
    } else {
      return HGCScintillatorDetId(type(), layer(), iradius(), iphi(), false);
    }
  }

  /// get the subdetector
  constexpr DetId::Detector subdet() const { return det(); }

  /// get/set the type
  constexpr int type() const { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }
  constexpr void setType(int type) {
    id_ &= kHGCalTypeMask0;
    id_ |= ((type & kHGCalTypeMask) << kHGCalTypeOffset);
  }
  constexpr int granularity() const { return (id_ >> kHGCalGranularityOffset) & kHGCalGranularityMask; }
  constexpr void setGranularity(int granularity) {
    id_ &= kHGCalGranularityMask0;
    id_ |= ((granularity & kHGCalGranularityMask) << kHGCalGranularityOffset);
  }

  /// get the z-side of the cell (1/-1)
  constexpr int zside() const { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }

  /// get the layer #
  constexpr int layer() const { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }

  /// get the eta index
  constexpr int ring() const {
    if (trigger())
      return (2 * ((id_ >> kHGCalRadiusOffset) & kHGCalRadiusMask));
    else
      return ((id_ >> kHGCalRadiusOffset) & kHGCalRadiusMask);
  }
  constexpr int iradiusAbs() const { return ring(); }
  constexpr int iradius() const { return zside() * ring(); }
  constexpr int ietaAbs() const { return ring(); }
  constexpr int ieta() const { return zside() * ring(); }

  /// get the phi index
  constexpr int iphi() const {
    if (trigger())
      return (2 * ((id_ >> kHGCalPhiOffset) & kHGCalPhiMask));
    else
      return ((id_ >> kHGCalPhiOffset) & kHGCalPhiMask);
  }
  constexpr std::pair<int, int> ietaphi() const { return std::pair<int, int>(ieta(), iphi()); }
  constexpr std::pair<int, int> ringphi() const { return std::pair<int, int>(iradius(), iphi()); }

  /// get/set the sipm size
  constexpr int sipm() const { return (id_ >> kHGCalSiPMOffset) & kHGCalSiPMMask; }
  constexpr void setSiPM(int sipm) {
    id_ &= kHGCalSiPMMask0;
    id_ |= ((sipm & kHGCalSiPMMask) << kHGCalSiPMOffset);
  }

  /// trigger or detector cell
  std::vector<HGCScintillatorDetId> detectorCells() const;

  constexpr bool trigger() const { return (((id_ >> kHGCalTriggerOffset) & kHGCalTriggerMask) == 1); }
  constexpr HGCScintillatorDetId triggerCell() const {
    if (trigger())
      return HGCScintillatorDetId(type(), layer(), iradius(), iphi(), true);
    else
      return HGCScintillatorDetId(type(), layer(), iradiusTrigger(), iphiTrigger(), true);
  }

  /// consistency check : no bits left => no overhead
  constexpr bool isEE() const { return false; }
  constexpr bool isHE() const { return true; }
  constexpr bool isForward() const { return true; }
  constexpr int position() const { return (id_ & kHGCalPositionMask); }

  static const HGCScintillatorDetId Undefined;

public:
  static constexpr int kHGCalPhiOffset = 0;
  static constexpr int kHGCalPhiMask = 0x1FF;
  static constexpr int kHGCalRadiusOffset = 9;
  static constexpr int kHGCalRadiusMask = 0xFF;
  static constexpr int kHGCalLayerOffset = 17;
  static constexpr int kHGCalLayerMask = 0x1F;
  static constexpr int kHGCalTriggerOffset = 22;
  static constexpr int kHGCalTriggerMask = 0x1;
  static constexpr int kHGCalSiPMOffset = 23;
  static constexpr int kHGCalSiPMMask = 0x1;
  static constexpr int kHGCalGranularityOffset = 24;
  static constexpr int kHGCalGranularityMask = 0x1;
  static constexpr int kHGCalGranularityMask0 = 0xFFDFFFFF;
  static constexpr int kHGCalSiPMMask0 = 0xFF7FFFFF;
  static constexpr int kHGCalZsideOffset = 25;
  static constexpr int kHGCalZsideMask = 0x1;
  static constexpr int kHGCalTypeOffset = 26;
  static constexpr int kHGCalTypeMask = 0x3;
  static constexpr int kHGCalTypeMask0 = 0xF3FFFFFF;
  static constexpr int kHGCalPositionMask = 0xF2FFFFFF;

  constexpr int iradiusTriggerAbs() const {
    if (trigger())
      return ((ring() + 1) / 2);
    else
      return ring();
  }
  constexpr int iradiusTrigger() const { return zside() * iradiusTriggerAbs(); }
  constexpr int iphiTrigger() const {
    if (trigger())
      return ((iphi() + 1) / 2);
    else
      return iphi();
  }
};

std::ostream& operator<<(std::ostream&, const HGCScintillatorDetId& id);

#endif
