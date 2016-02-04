#ifndef MONTIMINGLED1TTDAT_H
#define MONTIMINGLED1TTDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLed1TTDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_TT_L1_DAT";}
   
};

#endif
