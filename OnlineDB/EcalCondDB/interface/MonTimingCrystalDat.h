#ifndef MONTIMINGXTALDAT_H
#define MONTIMINGXTALDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingCrystalDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_CRYSTAL_DAT";}
   
};

#endif
