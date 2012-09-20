#include "OnlineDB/EcalCondDB/interface/LMFClsDat.h"

LMFClsDat::LMFClsDat() : LMFColoredTable() {
  init();
}

LMFClsDat::LMFClsDat(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn) : 
  LMFColoredTable(env, conn) {
  init();
}

LMFClsDat::LMFClsDat(EcalDBConnection *c) : LMFColoredTable(c) {
  init();
}

LMFClsDat::LMFClsDat(std::string color) : 
  LMFColoredTable() {
  init();
  setColor(color);
}

LMFClsDat::LMFClsDat(int color) : 
  LMFColoredTable() {
  init();
  setColor(color);
}

LMFClsDat::LMFClsDat(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn,
			 std::string color) : 
  LMFColoredTable(env, conn) {
  init();
  setColor(color);
}

LMFClsDat::LMFClsDat(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn,
			 int color) : 
  LMFColoredTable(env, conn) {
  init();
  setColor(color);
}

LMFClsDat::LMFClsDat(EcalDBConnection *c, std::string color) : LMFColoredTable(c) {
  init();
  setColor(color);
}

LMFClsDat::LMFClsDat(EcalDBConnection *c, int color) : LMFColoredTable(c) {
  init();
  setColor(color);
}

void LMFClsDat::init() {
  m_className = "LMFClsDat";
  m_keys["LMF_IOV_ID_REF"] = 0;
  m_keys["MEAN"] = 1;
  m_keys["NORM"] = 2;
  m_keys["RMS"] = 3;
  m_keys["NEVT"] = 4;
  m_keys["ENORM"] = 5;
  m_keys["FLAG"] = 6;
  m_keys["FLAGNORM"] = 7;
  m_keys["VMIN"] = 8;
  m_keys["VMAX"] = 9;
  for (unsigned int i = 0; i < m_keys.size(); i++) {
    m_type.push_back("NUMBER");
  }
  m_system = 0;
  m_color = 0;
}

bool LMFClsDat::isValid() {
  bool ret = true;
  if ((getColor() != "BLUE") && (getColor() != "IR")) {
    m_Error += " Color not properly set [" + getColor() + "]";
    ret = false;
  }
  return ret;
}

LMFClsDat& LMFClsDat::setLMFRefRunIOVID(EcalLogicID &id, int v) {
  LMFDat::setData(id, "LMF_IOV_ID_REF", v);
  return *this;
}

LMFClsDat& LMFClsDat::setMean(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MEAN", v);
  return *this;
}

LMFClsDat& LMFClsDat::setNorm(EcalLogicID &id, float v) {
  LMFDat::setData(id, "NORM", v);
  return *this;
}

LMFClsDat& LMFClsDat::setENorm(EcalLogicID &id, float v) {
  LMFDat::setData(id, "ENORM", v);
  return *this;
}

LMFClsDat& LMFClsDat::setRMS(EcalLogicID &id, float v) {
  LMFDat::setData(id, "RMS", v);
  return *this;
}

LMFClsDat& LMFClsDat::setNevt(EcalLogicID &id, int v) {
  LMFDat::setData(id, "NEVT", v);
  return *this;
}

LMFClsDat& LMFClsDat::setFlag(EcalLogicID &id, int v) {
  LMFDat::setData(id, "FLAG", v);
  return *this;
}

LMFClsDat& LMFClsDat::setEFlag(EcalLogicID &id, float v) {
  LMFDat::setData(id, "EFLAG", v);
  return *this;
}

int LMFClsDat::getLMFRefRunIOVID(EcalLogicID &id) {
  return getData(id, "LMF_IOV_ID_REF");
}

float LMFClsDat::getMean(EcalLogicID &id) {
  return getData(id, "MEAN");
}

float LMFClsDat::getNorm(EcalLogicID &id) {
  return getData(id, "NORM");
}

float LMFClsDat::getENorm(EcalLogicID &id) {
  return getData(id, "ENORM");
}

float LMFClsDat::getRMS(EcalLogicID &id) {
  return getData(id, "RMS");
}

int LMFClsDat::getNevt(EcalLogicID &id) {
  return getData(id, "NEVT");
}

int LMFClsDat::getFlag(EcalLogicID &id) {
  return getData(id, "FLAG");
}

float LMFClsDat::getEFlag(EcalLogicID &id) {
  return getData(id, "EFLAG");
}


