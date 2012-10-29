#ifndef LMFPRIMDAT_H
#define LMFPRIMDAT_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFColoredTable.h"

#include <math.h>

/**
 *   LMF_YYYY_XXX_PPRIM_DAT interface
 *        ^    ^
 *        |    |
 *        |    \_____ color
 *        \---------- system
 */
class LMFPrimDat : public LMFColoredTable {
 public:
  LMFPrimDat();
  LMFPrimDat(oracle::occi::Environment* env,
	      oracle::occi::Connection* conn);
  LMFPrimDat(EcalDBConnection *c);
  LMFPrimDat(std::string color, std::string system);
  LMFPrimDat(int color, std::string system);
  LMFPrimDat(oracle::occi::Environment* env,
	      oracle::occi::Connection* conn, std::string color,
	      std::string system);
  LMFPrimDat(EcalDBConnection *c, std::string color, std::string system);
  LMFPrimDat(oracle::occi::Environment* env,
	      oracle::occi::Connection* conn, int color,
	      std::string system);
  LMFPrimDat(EcalDBConnection *c, int color, std::string system);
  ~LMFPrimDat() {}

  std::string getTableName() const {
    return "LMF_" + getSystem() + "_" + getColor() + "_PRIM_DAT";
  }

  LMFPrimDat& setFlag(EcalLogicID &id, int v);
  LMFPrimDat& setMean(EcalLogicID &id, float v);
  LMFPrimDat& setRMS(EcalLogicID &id, float v);
  LMFPrimDat& setM3(EcalLogicID &id, float v);
  LMFPrimDat& setPN(EcalLogicID &id, float mean, float rms, float m3);
  LMFPrimDat& setAPDoverAMean(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverARMS(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverAM3(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverA(EcalLogicID &id, float mean, float rms, float m3);
  LMFPrimDat& setAPDoverBMean(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverBRMS(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverBM3(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverB(EcalLogicID &id, float mean, float rms, float m3);
  LMFPrimDat& setAPDoverPnMean(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverPnRMS(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverPnM3(EcalLogicID &id, float v);
  LMFPrimDat& setAPDoverPn(EcalLogicID &id, float mean, float rms, float m3);
  LMFPrimDat& setAlpha(EcalLogicID &id, float v);
  LMFPrimDat& setBeta(EcalLogicID &id, float v);
  LMFPrimDat& setShapeCorr(EcalLogicID &id, float v);

  float getMean(EcalLogicID &id);
  float getRMS(EcalLogicID &id);
  float getM3(EcalLogicID &id);
  int   getFlag(EcalLogicID &id);
  float getAPDoverAMean(EcalLogicID &id);
  float getAPDoverARMS(EcalLogicID &id);
  float getAPDoverAM3(EcalLogicID &id);
  float getAPDoverBMean(EcalLogicID &id);
  float getAPDoverBRMS(EcalLogicID &id);
  float getAPDoverBM3(EcalLogicID &id);
  float getAPDoverPnMean(EcalLogicID &id);
  float getAPDoverPnRMS(EcalLogicID &id);
  float getAPDoverPnM3(EcalLogicID &id);
  float getAlpha(EcalLogicID &id);
  float getBeta(EcalLogicID &id);
  float getShapeCorr(EcalLogicID &id);

  bool isValid();
  // to do: complete list of set/get methods

 private:
  void init();
};

#endif
