#ifndef MONTIMINGLASERREDXTALDAT_H
#define MONTIMINGLASERREDXTALDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLaserRedCrystalDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_XTAL_LR_DAT";}
   
};

#endif
