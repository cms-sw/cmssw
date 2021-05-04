#ifndef DataFormats_MuonDetId_GEMSubDetId_h
#define DataFormats_MuonDetId_GEMSubDetId_h

/** \class GEMSubDetId
 *
 */

#include <cstdint>

class GEMSubDetId {
public:
  enum class Station { GE0 = 0, ME0 = 0, GE11 = 1, GE21 = 2 };
  static Station station(uint16_t st) {
    Station returnValue = Station::GE11;
    if (st == 0) {
      returnValue = Station::ME0;
    } else if (st == 2) {
      returnValue = Station::GE21;
    }
    return returnValue;
  };
};

#endif
