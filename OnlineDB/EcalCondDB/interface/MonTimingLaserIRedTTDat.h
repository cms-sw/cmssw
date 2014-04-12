#ifndef MONTIMINGLASERIRTTDAT_H
#define MONTIMINGLASERIRTTDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLaserIRedTTDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_TT_LI_DAT";}
   
};

#endif
