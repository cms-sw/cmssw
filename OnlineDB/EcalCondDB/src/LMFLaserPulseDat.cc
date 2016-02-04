#include "OnlineDB/EcalCondDB/interface/LMFLaserPulseDat.h"

LMFLaserPulseDat::LMFLaserPulseDat() : LMFColoredTable() {
  init();
}

LMFLaserPulseDat::LMFLaserPulseDat(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn) : 
  LMFColoredTable(env, conn) {
  init();
}

LMFLaserPulseDat::LMFLaserPulseDat(EcalDBConnection *c) : LMFColoredTable(c) {
  init();
}

LMFLaserPulseDat::LMFLaserPulseDat(std::string color) : 
  LMFColoredTable() {
  init();
  setColor(color);
}

LMFLaserPulseDat::LMFLaserPulseDat(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn,
			 std::string color) : 
  LMFColoredTable(env, conn) {
  init();
  setColor(color);
}

LMFLaserPulseDat::LMFLaserPulseDat(EcalDBConnection *c, std::string color) : 
  LMFColoredTable(c) {
  init();
  setColor(color);
}

void LMFLaserPulseDat::init() {
  m_className = "LMFLaserPulseDat";
  m_keys["FIT_METHOD"] = 0;
  m_keys["MTQ_AMPL"]   = 1;
  m_keys["MTQ_TIME"]   = 2;
  m_keys["MTQ_RISE"]   = 3;
  m_keys["MTQ_FWHM"]   = 4;
  m_keys["MTQ_FW20"]   = 5;
  m_keys["MTQ_FW80"]   = 6;
  m_keys["MTQ_SLIDING"]= 7;
  m_keys["VMIN"]       = 8;
  m_keys["VMAX"]       = 9;
  for (int i = 0; i < 10; i++) {
    m_type.push_back("NUMBER");
  }
  setSystem("LASER");
  m_color = 0;
}

bool LMFLaserPulseDat::isValid() {
  bool ret = true;
  ret = LMFDat::isValid();
  if ((getColor() != "BLUE") && (getColor() != "IR")) {
    m_Error += " Color not properly set [" + getColor() + "]";
    ret = false;
  }
  return ret;
}

LMFLaserPulseDat& LMFLaserPulseDat::setFitMethod(EcalLogicID &id, int v) {
  LMFDat::setData(id, "FIT_METHOD", v);
  return *this;
}

LMFLaserPulseDat& LMFLaserPulseDat::setMTQAmplification(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MTQ_AMPL", v);
  return *this;
}

LMFLaserPulseDat& LMFLaserPulseDat::setMTQTime(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MTQ_TIME", v);
  return *this;
}

LMFLaserPulseDat& LMFLaserPulseDat::setMTQRise(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MTQ_RISE", v);
  return *this;
}

LMFLaserPulseDat& LMFLaserPulseDat::setMTQFWHM(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MTQ_FWHM", v);
  return *this;
}

LMFLaserPulseDat& LMFLaserPulseDat::setMTQFW20(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MTQ_FW20", v);
  return *this;
}

LMFLaserPulseDat& LMFLaserPulseDat::setMTQFW80(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MTQ_FW80", v);
  return *this;
}

LMFLaserPulseDat& LMFLaserPulseDat::setMTQSliding(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MTQ_SLIDING", v); 
  return *this;
}

int LMFLaserPulseDat::getFitMethod(EcalLogicID &id) {
  return getData(id, "FIT_METHOD");
}

float LMFLaserPulseDat::getMTQAmplification(EcalLogicID &id) {
  return getData(id, "MTQ_AMPL");
}

float LMFLaserPulseDat::getMTQTime(EcalLogicID &id) {
  return getData(id, "MTQ_TIME");
}

float LMFLaserPulseDat::getMTQRise(EcalLogicID &id) {
  return getData(id, "MTQ_RISE");
}

float LMFLaserPulseDat::getMTQFWHM(EcalLogicID &id) {
  return getData(id, "MTQ_FWHM");
}

float LMFLaserPulseDat::getMTQFW20(EcalLogicID &id) {
  return getData(id, "MTQ_FW20");
}

float LMFLaserPulseDat::getMTQFW80(EcalLogicID &id) {
  return getData(id, "MTQ_FW80");
}

float LMFLaserPulseDat::getMTQSliding(EcalLogicID &id) {
  return getData(id, "MTQ_SLIDING");
}

