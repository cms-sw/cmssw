#ifndef MONTIMINGLED2TTDAT_H
#define MONTIMINGLED2TTDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLed2TTDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_TT_L2_DAT";}
   
};

#endif
