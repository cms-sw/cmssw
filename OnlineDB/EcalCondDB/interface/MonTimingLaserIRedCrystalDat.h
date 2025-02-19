#ifndef MONTIMINGLASERIREDXTALDAT_H
#define MONTIMINGLASERIREDXTALDAT_H

#include "OnlineDB/EcalCondDB/interface/ITimingDat.h"

class MonTimingLaserIRedCrystalDat : public ITimingDat {
 public:
  // User data methods
  inline std::string getTable() { return "MON_TIMING_XTAL_LI_DAT";}
   
};

#endif
