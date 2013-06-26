#ifndef MONTIMINGLASERGREENTTDAT_H
#define MONTIMINGLASERGREENTTDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLaserGreenTTDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_TT_LG_DAT";}
   
};

#endif
