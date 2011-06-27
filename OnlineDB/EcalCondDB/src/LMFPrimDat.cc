#include "OnlineDB/EcalCondDB/interface/LMFPrimDat.h"

LMFPrimDat::LMFPrimDat() : LMFColoredTable() {
  init();
}

LMFPrimDat::LMFPrimDat(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn) : 
  LMFColoredTable(env, conn) {
  init();
}

LMFPrimDat::LMFPrimDat(EcalDBConnection *c) : LMFColoredTable(c) {
  init();
}

LMFPrimDat::LMFPrimDat(std::string color, std::string system) : 
  LMFColoredTable() {
  init();
  setColor(color);
  setSystem(system);
}

LMFPrimDat::LMFPrimDat(int color, std::string system) : 
  LMFColoredTable() {
  init();
  setColor(color);
  setSystem(system);
}

LMFPrimDat::LMFPrimDat(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn,
			 std::string color, std::string system) : 
  LMFColoredTable(env, conn) {
  init();
  setColor(color);
  setSystem(system);
}

LMFPrimDat::LMFPrimDat(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn,
			 int color, std::string system) : 
  LMFColoredTable(env, conn) {
  init();
  setColor(color);
  setSystem(system);
}

LMFPrimDat::LMFPrimDat(EcalDBConnection *c, std::string color, 
			 std::string system) : LMFColoredTable(c) {
  init();
  setColor(color);
  setSystem(system);
}

LMFPrimDat::LMFPrimDat(EcalDBConnection *c, int color, 
			 std::string system) : LMFColoredTable(c) {
  init();
  setColor(color);
  setSystem(system);
}

void LMFPrimDat::init() {
  m_className = "LMFPrimDat";
  m_keys["FLAG"] = 0;
  m_keys["MEAN"] = 1;
  m_keys["RMS"] = 2;
  m_keys["M3"] = 3;
  m_keys["APDAMEAN"] = 4;
  m_keys["APDARMS"] = 5;
  m_keys["APDAM3"] = 6;
  m_keys["APDBMEAN"] = 7;
  m_keys["APDBRMS"] = 8;
  m_keys["APDBM3"] = 9;
  m_keys["APDPNMEAN"] = 10;
  m_keys["APDPNRMS"] = 11;
  m_keys["APDPNM3"] = 12;
  m_keys["ALPHA"] = 13;
  m_keys["BETA"] = 14;
  m_keys["SHAPECORR"] = 15;
  m_keys["VMIN"] = 16;
  m_keys["VMAX"] = 17;
  for (unsigned int i = 0; i < m_keys.size(); i++) {
    m_type.push_back("NUMBER");
  }
  m_system = 0;
  m_color = 0;
}

bool LMFPrimDat::isValid() {
  bool ret = true;
  if ((getSystem() != "LASER") && (getSystem() != "LED")) {
    m_Error += " System name not properly set [" + getSystem() + "]";
    ret = false;
  }
  if ((getSystem() == "LASER") && 
      (getColor() != "BLUE") && (getColor() != "IR")) {
    m_Error += " Color not properly set [" + getColor() + "]";
    ret = false;
  }
  if ((getSystem() == "LED") && 
      (getColor() != "BLUE") && (getColor() != "ORANGE")) {
    m_Error += " Color not properly set [" + getColor() + "]";
    ret = false;
  }
  return ret;
}

LMFPrimDat& LMFPrimDat::setMean(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MEAN", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setRMS(EcalLogicID &id, float v) {
  LMFDat::setData(id, "RMS", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setM3(EcalLogicID &id, float v) {
  LMFDat::setData(id, "M3", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setPN(EcalLogicID &id, float mean, float rms, 
				  float m3) {
  setMean(id, mean);
  setRMS(id, rms);
  setM3(id, m3);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverAM3(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDAM3", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverAMean(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDAMEAN", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverARMS(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDARMS", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverA(EcalLogicID &id,
				    float mean, float rms, float m3) {
  setAPDoverAMean(id, mean);
  setAPDoverARMS(id, rms);
  setAPDoverAM3(id, m3);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverBM3(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDBM3", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverBMean(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDBMEAN", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverBRMS(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDBRMS", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverB(EcalLogicID &id, 
					float mean, float rms, float m3) {
  setAPDoverBMean(id, mean);
  setAPDoverBRMS(id, rms);
  setAPDoverBM3(id, m3);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverPnM3(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDPNM3", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverPnMean(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDPNMEAN", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverPnRMS(EcalLogicID &id, float v) {
  LMFDat::setData(id, "APDPNRMS", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAPDoverPn(EcalLogicID &id,
				    float mean, float rms, float m3) {
  setAPDoverPnMean(id, mean);
  setAPDoverPnRMS(id, rms);
  setAPDoverPnM3(id, m3);
  return *this;
}

LMFPrimDat& LMFPrimDat::setFlag(EcalLogicID &id, int v) {
  LMFDat::setData(id, "FLAG", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setAlpha(EcalLogicID &id, float v) {
  LMFDat::setData(id, "ALPHA", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setBeta(EcalLogicID &id, float v) {
  LMFDat::setData(id, "BETA", v);
  return *this;
}

LMFPrimDat& LMFPrimDat::setShapeCorr(EcalLogicID &id, float v) {
  LMFDat::setData(id, "SHAPECORR", v);
  return *this;
}

float LMFPrimDat::getMean(EcalLogicID &id) {
  return getData(id, "MEAN");
}

float LMFPrimDat::getRMS(EcalLogicID &id) {
  return getData(id, "RMS");
}

float LMFPrimDat::getM3(EcalLogicID &id) {
  return getData(id, "M3");
}

float LMFPrimDat::getAPDoverAM3(EcalLogicID &id) {
  return getData(id, "APDAM3");
}

float LMFPrimDat::getAPDoverAMean(EcalLogicID &id) {
  return getData(id, "APDAMEAN");
}

float LMFPrimDat::getAPDoverARMS(EcalLogicID &id) {
  return getData(id, "APDARMS");
}

float LMFPrimDat::getAPDoverBM3(EcalLogicID &id) {
  return getData(id, "APDBM3");
}

float LMFPrimDat::getAPDoverPnMean(EcalLogicID &id) {
  return getData(id, "APDPNMEAN");
}

float LMFPrimDat::getAPDoverPnRMS(EcalLogicID &id) {
  return getData(id, "APDPNRMS");
}

float LMFPrimDat::getAPDoverPnM3(EcalLogicID &id) {
  return getData(id, "APDPNM3");
}

float LMFPrimDat::getAPDoverBMean(EcalLogicID &id) {
  return getData(id, "APDBMEAN");
}

float LMFPrimDat::getAPDoverBRMS(EcalLogicID &id) {
  return getData(id, "APDBRMS");
}

float LMFPrimDat::getAlpha(EcalLogicID &id) {
  return getData(id, "ALPHA");
}

float LMFPrimDat::getBeta(EcalLogicID &id) {
  return getData(id, "BETA");
}

float LMFPrimDat::getShapeCorr(EcalLogicID &id) {
  return getData(id, "SHAPECORR");
}

int LMFPrimDat::getFlag(EcalLogicID &id) {
  return getData(id, "FLAG");
}

