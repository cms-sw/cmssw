#ifndef MONTIMINGLED2XTALDAT_H
#define MONTIMINGLED2XTALDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLed2CrystalDat : public ITimingDat {
public:
  // User data methods
  inline std::string getTable() override { return "MON_TIMING_XTAL_L2_DAT"; }
};

#endif
