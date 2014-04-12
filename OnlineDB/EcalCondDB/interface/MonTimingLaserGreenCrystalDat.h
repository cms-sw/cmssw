#ifndef MONTIMINGLASERGREENXTALDAT_H
#define MONTIMINGLASERGREENXTALDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLaserGreenCrystalDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_XTAL_LG_DAT";}
   
};

#endif
