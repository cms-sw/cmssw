#ifndef MONTIMINGLASERBLUETTDAT_H
#define MONTIMINGLASERBLUETTDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLaserBlueTTDat : public ITimingDat {
public:
  // User data methods
  inline std::string getTable() override { return "MON_TIMING_TT_LB_DAT"; }
};

#endif
