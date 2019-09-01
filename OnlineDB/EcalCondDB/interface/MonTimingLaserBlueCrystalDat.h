#ifndef MONTIMINGLASERBLUEXTALDAT_H
#define MONTIMINGLASERBLUEXTALDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLaserBlueCrystalDat : public ITimingDat {
  friend class ITimingDat;

public:
  // User data methods
  inline std::string getTable() override { return "MON_TIMING_XTAL_LB_DAT"; }
};

#endif
