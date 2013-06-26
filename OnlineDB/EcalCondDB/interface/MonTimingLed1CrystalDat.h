#ifndef MONTIMINGLED1XTALDAT_H
#define MONTIMINGLED1XTALDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLed1CrystalDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_XTAL_L1_DAT";}
   
};

#endif
