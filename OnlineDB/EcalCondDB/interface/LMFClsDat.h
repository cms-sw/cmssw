#ifndef LMFCLSDAT_H
#define LMFCLSDAT_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFColoredTable.h"

#include <cmath>

/**
 *   LMF_CLS_XXXX_DAT interface
 *            ^
 *            |
 *            \_____ color
 */
class LMFClsDat : public LMFColoredTable {
public:
  typedef oracle::occi::ResultSet ResultSet;
  typedef oracle::occi::Statement Statement;

  LMFClsDat();
  LMFClsDat(oracle::occi::Environment *env, oracle::occi::Connection *conn);
  LMFClsDat(EcalDBConnection *c);
  LMFClsDat(std::string color);
  LMFClsDat(int color);
  LMFClsDat(oracle::occi::Environment *env, oracle::occi::Connection *conn, std::string color);
  LMFClsDat(EcalDBConnection *c, std::string color);
  LMFClsDat(oracle::occi::Environment *env, oracle::occi::Connection *conn, int color);
  LMFClsDat(EcalDBConnection *c, int color);
  ~LMFClsDat() override {}

  std::string getTableName() const override { return "LMF_CLS_" + getColor() + "_DAT"; }

  LMFClsDat &setSystem(int system) override { return *this; }
  LMFClsDat &setSystem(std::string system) override { return *this; }

  LMFClsDat &setLMFRefRunIOVID(EcalLogicID &id, int v);
  LMFClsDat &setMean(EcalLogicID &id, float v);
  LMFClsDat &setNorm(EcalLogicID &id, float v);
  LMFClsDat &setENorm(EcalLogicID &id, float v);
  LMFClsDat &setRMS(EcalLogicID &id, float v);
  LMFClsDat &setNevt(EcalLogicID &id, int v);
  LMFClsDat &setFlag(EcalLogicID &id, int v);
  LMFClsDat &setFlagNorm(EcalLogicID &id, float v);

  int getLMFRefRunIOVID(EcalLogicID &id);
  float getMean(EcalLogicID &id);
  float getNorm(EcalLogicID &id);
  float getENorm(EcalLogicID &id);
  float getRMS(EcalLogicID &id);
  int getNevt(EcalLogicID &id);
  int getFlag(EcalLogicID &id);
  float getFlagNorm(EcalLogicID &id);

  std::string getSystem() const override { return ""; }

  bool isValid() override;
  // to do: complete list of set/get methods

private:
  void init();
};

#endif
