#ifndef MONTIMINGLASERREDXTALDAT_H
#define MONTIMINGLASERREDXTALDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLaserRedTTDat : public ITimingDat {
public:
  // User data methods
  inline std::string getTable() override { return "MON_TIMING_TT_LR_DAT"; }
};

#endif
