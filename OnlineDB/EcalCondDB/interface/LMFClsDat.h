#ifndef LMFCLSDAT_H
#define LMFCLSDAT_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFColoredTable.h"

#include <math.h>

/**
 *   LMF_CLS_XXXX_DAT interface
 *            ^
 *            |
 *            \_____ color
 */
class LMFClsDat : public LMFColoredTable {
 public:
  LMFClsDat();
  LMFClsDat(oracle::occi::Environment* env,
	    oracle::occi::Connection* conn);
  LMFClsDat(EcalDBConnection *c);
  LMFClsDat(std::string color);
  LMFClsDat(int color);
  LMFClsDat(oracle::occi::Environment* env,
	    oracle::occi::Connection* conn, std::string color);
  LMFClsDat(EcalDBConnection *c, std::string color);
  LMFClsDat(oracle::occi::Environment* env,
	    oracle::occi::Connection* conn, int color);
  LMFClsDat(EcalDBConnection *c, int color);
  ~LMFClsDat() {}

  std::string getTableName() {
    return "LMF_CLS_" + getColor() + "_DAT";
  }
  
  LMFClsDat& setSystem(int system) { return *this; }
  LMFClsDat& setSystem(std::string system) { return *this; }

  LMFClsDat& setLMFRefRunIOV(const LMFRunIOV &iov) {
    setInt("lmfRunIOVRef_id", iov.getID());
    attach("lmfRunIOVRef", (LMFUnique*)&iov);
    return *this;
  }

  LMFClsDat& setMean(EcalLogicID &id, float v);
  LMFClsDat& setNorm(EcalLogicID &id, float v);
  LMFClsDat& setENorm(EcalLogicID &id, float v);
  LMFClsDat& setRMS(EcalLogicID &id, float v);
  LMFClsDat& setNevt(EcalLogicID &id, int v);
  LMFClsDat& setFlag(EcalLogicID &id, int v);
  LMFClsDat& setEFlag(EcalLogicID &id, float v);

  float getMean(EcalLogicID &id);
  float getNorm(EcalLogicID &id);
  float getENorm(EcalLogicID &id);
  float getRMS(EcalLogicID &id);
  int   getNevt(EcalLogicID &id);
  int   getFlag(EcalLogicID &id);
  float getEFlag(EcalLogicID &id);

  std::string getSystem() { return ""; }

  LMFRunIOV getLMFRefRunIOV() const {
    LMFRunIOV runiov(m_env, m_conn);
    runiov.setByID(getInt("lmfRunIOVRef_id"));
    return runiov;
  }

  bool isValid();
  // to do: complete list of set/get methods

 private:
  void init();
};

#endif
