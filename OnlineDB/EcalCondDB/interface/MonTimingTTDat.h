#ifndef MONTIMINGTTDAT_H
#define MONTIMINGTTDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingTTDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_TT_DAT";}
   
};

#endif
