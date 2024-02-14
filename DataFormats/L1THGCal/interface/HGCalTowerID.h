#ifndef DataFormats_L1THGCal_HGCalTowerID_h
#define DataFormats_L1THGCal_HGCalTowerID_h

#include <cstdint>

// NOTE: in the current implementation HGCalTowerID can only
// accomodate 127 bins per coordinate x2 zsides

namespace l1t {
  class HGCalTowerID {
  public:
    HGCalTowerID() : HGCalTowerID(0) {}

    HGCalTowerID(uint32_t rawId) : rawId_(rawId) {}

    HGCalTowerID(short subdetIsNode, short zside, unsigned short coord1, unsigned short coord2) {
      rawId_ = (((subdetIsNode & subDetMask) << subDetShift) | ((coord1 & coordMask) << coord1Shift) |
                ((coord2 & coordMask) << coord2Shift) | ((zside > 0) & zsideMask) << zsideShift);
    }

    short subdet() const { return (rawId_ >> subDetShift) & subDetMask; }

    short zside() const { return ((rawId_ >> zsideShift) & zsideMask) ? 1 : -1; }

    unsigned short iEta() const { return (rawId_ >> coord1Shift) & coordMask; }

    unsigned short iPhi() const { return (rawId_ >> coord2Shift) & coordMask; }

    unsigned short rawId() const { return rawId_; }

    static const int subDetMask = 0x1;  // two for now 0 is HGC and 1 is HFNose
    static const int subDetShift = 16;
    static const int zsideMask = 0x1;
    static const int zsideShift = 15;
    static const int coordMask = 0x007F;
    static const int coord1Shift = 7;
    static const int coord2Shift = 0;

  private:
    uint32_t rawId_;
  };

  struct HGCalTowerCoord {
    HGCalTowerCoord(uint32_t rawId, float eta, float phi) : rawId(rawId), eta(eta), phi(phi) {}

    const uint32_t rawId;
    const float eta;
    const float phi;
  };
}  // namespace l1t

#endif
