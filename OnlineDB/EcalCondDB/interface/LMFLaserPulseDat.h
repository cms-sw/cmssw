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

  int getFitMethod(EcalLogicID &id);
  float getMTQAmplification(EcalLogicID &id);
  float getMTQTime(EcalLogicID &id);
  float getMTQRise(EcalLogicID &id);
  float getMTQFWHM(EcalLogicID &id);
  float getMTQFW20(EcalLogicID &id);
  float getMTQFW80(EcalLogicID &id);
  float getMTQSliding(EcalLogicID &id);

  bool isValid();
  // to do: complete list of set/get methods

 private:
  void init();
};

#endif
