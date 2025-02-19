#ifndef LMFPULSEDAT_H
#define LMFPULSEDAT_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFColoredTable.h"

#include <math.h>

/**
 *   LMF_LASER_XXX_PULSE_DAT interface
 *        ^    ^
 *        |    |
 *        |    \_____ color
 *        \---------- system
 */
class LMFLaserPulseDat : public LMFColoredTable {
 public:
  LMFLaserPulseDat();
  LMFLaserPulseDat(oracle::occi::Environment* env,
		   oracle::occi::Connection* conn);
  LMFLaserPulseDat(EcalDBConnection *c);
  LMFLaserPulseDat(std::string color);
  LMFLaserPulseDat(int color);
  LMFLaserPulseDat(oracle::occi::Environment* env,
		   oracle::occi::Connection* conn, std::string color);
  LMFLaserPulseDat(oracle::occi::Environment* env,
		   oracle::occi::Connection* conn, int color);
  LMFLaserPulseDat(EcalDBConnection *c, std::string color);
  LMFLaserPulseDat(EcalDBConnection *c, int color);
  ~LMFLaserPulseDat() {}
  
  std::string getTableName() const {
    return "LMF_LASER_" + getColor() + "_PULSE_DAT";
  }
  
  LMFLaserPulseDat& setFitMethod(EcalLogicID &id, int v);
  LMFLaserPulseDat& setMTQAmplification(EcalLogicID &id, float v);
  LMFLaserPulseDat& setMTQTime(EcalLogicID &id, float v);
  LMFLaserPulseDat& setMTQRise(EcalLogicID &id, float v);
  LMFLaserPulseDat& setMTQFWHM(EcalLogicID &id, float v);
  LMFLaserPulseDat& setMTQFW20(EcalLogicID &id, float v);
  LMFLaserPulseDat& setMTQFW80(EcalLogicID &id, float v);
  LMFLaserPulseDat& setMTQSliding(EcalLogicID &id, float v);

  int getFitMethod(EcalLogicID &id) { return getFitMethod(id.getLogicID()); }
  float getMTQAmplification(EcalLogicID &id) { 
    return getMTQAmplification(id.getLogicID()); 
  }
  float getMTQTime(EcalLogicID &id) { return getMTQTime(id.getLogicID()); }
  float getMTQRise(EcalLogicID &id) { return getMTQRise(id.getLogicID()); }
  float getMTQFWHM(EcalLogicID &id) { return getMTQFWHM(id.getLogicID()); }
  float getMTQFW20(EcalLogicID &id) { return getMTQFW20(id.getLogicID()); }
  float getMTQFW80(EcalLogicID &id) { return getMTQFW80(id.getLogicID()); }
  float getMTQSliding(EcalLogicID &id) { 
    return getMTQSliding(id.getLogicID()); 
  }
  int getFitMethod(int id);
  float getMTQAmplification(int id);
  float getMTQTime(int id);
  float getMTQRise(int id);
  float getMTQFWHM(int id);
  float getMTQFW20(int id);
  float getMTQFW80(int id);
  float getMTQSliding(int id);

  bool isValid();
  // to do: complete list of set/get methods

 private:
  void init();
};

#endif
