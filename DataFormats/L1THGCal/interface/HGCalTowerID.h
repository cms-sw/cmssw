#ifndef DataFormats_L1TCalorimeter_HGCalTowerID_h
#define DataFormats_L1TCalorimeter_HGCalTowerID_h

namespace l1t {
  class HGCalTowerID {
  public:
    HGCalTowerID(unsigned short rawId): rawId_(rawId) {}

    HGCalTowerID(short zside, unsigned short iX, unsigned short iY) {
      rawId_ = ((iX & coordMask) << xShift) | ((iY & coordMask) << yShift) | (((zside > 0) & zsideMask) << zsideShift);
    }

    short zside() const { return ((rawId_ >> zsideShift) & zsideMask) ? 1 : -1;}

    unsigned short iX() const { return (rawId_ >> xShift) & coordMask; }

    unsigned short iY() const { return (rawId_ >> yShift) & coordMask;}

    unsigned short rawId() const {return rawId_;}


  private:

    unsigned short rawId_;
    static const int zsideMask = 0x1;
    static const int zsideShift = 15;
    static const int coordMask = 0x007F;
    static const int xShift = 7;
    static const int yShift = 0;
  };
}


#endif
