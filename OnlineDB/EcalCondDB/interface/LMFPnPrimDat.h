#ifndef LMFPNPRIMDAT_H
#define LMFPNPRIMDAT_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFColoredTable.h"

#include <cmath>

/**
 *   LMF_YYYY_XXX_PN_PRIM_DAT interface
 *        ^    ^
 *        |    |
 *        |    \_____ color
 *        \---------- system
 */
class LMFPnPrimDat : public LMFColoredTable {
public:
  LMFPnPrimDat();
  LMFPnPrimDat(oracle::occi::Environment *env, oracle::occi::Connection *conn);
  LMFPnPrimDat(EcalDBConnection *c);
  LMFPnPrimDat(std::string color, std::string system);
  LMFPnPrimDat(oracle::occi::Environment *env, oracle::occi::Connection *conn, std::string color, std::string system);
  LMFPnPrimDat(EcalDBConnection *c, std::string color, std::string system);
  LMFPnPrimDat(EcalDBConnection *c, std::string color, std::string system, bool debug);
  LMFPnPrimDat(int color, std::string system);
  LMFPnPrimDat(oracle::occi::Environment *env, oracle::occi::Connection *conn, int color, std::string system);
  LMFPnPrimDat(EcalDBConnection *c, int color, std::string system);
  ~LMFPnPrimDat() override {}

  std::string getTableName() const override { return "LMF_" + getSystem() + "_" + getColor() + "_PN_PRIM_DAT"; }

  LMFPnPrimDat &setMean(EcalLogicID &id, float v);
  LMFPnPrimDat &setRMS(EcalLogicID &id, float v);
  LMFPnPrimDat &setM3(EcalLogicID &id, float v);
  LMFPnPrimDat &setPN(EcalLogicID &id, float mean, float rms, float m3);
  LMFPnPrimDat &setShapeCorr(EcalLogicID &id, float mean);
  LMFPnPrimDat &setPNAoverBMean(EcalLogicID &id, float v);
  LMFPnPrimDat &setPNAoverBRMS(EcalLogicID &id, float v);
  LMFPnPrimDat &setPNAoverBM3(EcalLogicID &id, float v);
  LMFPnPrimDat &setPNAoverB(EcalLogicID &id, float mean, float rms, float m3);
  LMFPnPrimDat &setFlag(EcalLogicID &id, int v);

  LMFPnPrimDat &setSystem(std::string s) override;

  float getMean(EcalLogicID &id);
  float getRMS(EcalLogicID &id);
  float getM3(EcalLogicID &id);
  float getPNAoverBMean(EcalLogicID &id);
  float getPNAoverBRMS(EcalLogicID &id);
  float getPNAoverBM3(EcalLogicID &id);
  float getShapeCor(EcalLogicID &id);
  int getFlag(EcalLogicID &id);

  float getMean(int id);
  float getRMS(int id);
  float getM3(int id);
  float getPNAoverBMean(int id);
  float getPNAoverBRMS(int id);
  float getPNAoverBM3(int id);
  float getShapeCor(int id);
  int getFlag(int id);

  bool isValid() override;
  // to do: complete list of set/get methods

private:
  void init();
};

#endif
