#ifndef DataFormats_L1TCalorimeter_HGCalTowerID_h
#define DataFormats_L1TCalorimeter_HGCalTowerID_h

// NOTE: in the current implementation HGCalTowerID can only
// accomodate 127 bins per coordinate x2 zsides

namespace l1t {
  class HGCalTowerID {
  public:
    HGCalTowerID() : HGCalTowerID(0) {}

    HGCalTowerID(unsigned short rawId) : rawId_(rawId) {}

    HGCalTowerID(short zside, unsigned short coord1, unsigned short coord2) {
      rawId_ = ((coord1 & coordMask) << coord1Shift) | ((coord2 & coordMask) << coord2Shift) |
               (((zside > 0) & zsideMask) << zsideShift);
    }

    short zside() const { return ((rawId_ >> zsideShift) & zsideMask) ? 1 : -1; }

    unsigned short iEta() const { return (rawId_ >> coord1Shift) & coordMask; }

    unsigned short iPhi() const { return (rawId_ >> coord2Shift) & coordMask; }

    unsigned short rawId() const { return rawId_; }

  private:
    unsigned short rawId_;
    static const int zsideMask = 0x1;
    static const int zsideShift = 15;
    static const int coordMask = 0x007F;
    static const int coord1Shift = 7;
    static const int coord2Shift = 0;
  };

  struct HGCalTowerCoord {
    HGCalTowerCoord(unsigned short rawId, float eta, float phi) : rawId(rawId), eta(eta), phi(phi) {}

    const unsigned short rawId;
    const float eta;
    const float phi;
  };
}  // namespace l1t

#endif
