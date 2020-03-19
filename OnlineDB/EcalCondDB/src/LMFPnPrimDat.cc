#include "OnlineDB/EcalCondDB/interface/LMFPnPrimDat.h"

LMFPnPrimDat::LMFPnPrimDat() : LMFColoredTable() { init(); }

LMFPnPrimDat::LMFPnPrimDat(oracle::occi::Environment *env, oracle::occi::Connection *conn)
    : LMFColoredTable(env, conn) {
  init();
}

LMFPnPrimDat::LMFPnPrimDat(EcalDBConnection *c) : LMFColoredTable(c) { init(); }

LMFPnPrimDat::LMFPnPrimDat(std::string color, std::string system) : LMFColoredTable() {
  init();
  setColor(color);
  setSystem(system);
}

LMFPnPrimDat::LMFPnPrimDat(oracle::occi::Environment *env,
                           oracle::occi::Connection *conn,
                           std::string color,
                           std::string system)
    : LMFColoredTable(env, conn) {
  init();
  setColor(color);
  setSystem(system);
}

LMFPnPrimDat::LMFPnPrimDat(EcalDBConnection *c, std::string color, std::string system, bool d) : LMFColoredTable(c) {
  if (d) {
    debug();
  }
  init();
  setColor(color);
  setSystem(system);
}

LMFPnPrimDat::LMFPnPrimDat(EcalDBConnection *c, std::string color, std::string system) : LMFColoredTable(c) {
  init();
  setColor(color);
  setSystem(system);
}

LMFPnPrimDat::LMFPnPrimDat(int color, std::string system) : LMFColoredTable() {
  init();
  setColor(color);
  setSystem(system);
}

LMFPnPrimDat::LMFPnPrimDat(oracle::occi::Environment *env, oracle::occi::Connection *conn, int color, std::string system)
    : LMFColoredTable(env, conn) {
  init();
  setColor(color);
  setSystem(system);
}

LMFPnPrimDat::LMFPnPrimDat(EcalDBConnection *c, int color, std::string system) : LMFColoredTable(c) {
  init();
  setColor(color);
  setSystem(system);
}

LMFPnPrimDat &LMFPnPrimDat::setSystem(std::string s) {
  // LED tables do not hold the shapecorr column. Drop it.
  std::transform(s.begin(), s.end(), s.begin(), toupper);
  if (s == "LED") {
    if (m_debug) {
      std::cout << "Erasing unwanted data" << std::endl;
    }
    m_type.erase(m_type.begin());
    m_keys.erase("SHAPECORRPN");
    if (m_debug) {
      std::cout << "Data: " << m_data.size() << " Keys: " << m_keys.size() << " Type: " << m_type.size() << std::endl;
    }
    std::map<std::string, unsigned int>::iterator i = m_keys.begin();
    std::map<std::string, unsigned int>::iterator e = m_keys.end();
    while (i != e) {
      // modify indexes
      (i->second)--;
      if (m_debug) {
        std::cout << "Key " << i->first << " = " << i->second << std::endl;
      }
      i++;
    }
  }
  LMFColoredTable::setSystem(s);
  return *this;
}

void LMFPnPrimDat::init() {
  m_className = "LMFPnPrimDat";

  m_keys["SHAPECORRPN"] = 0;
  m_keys["MEAN"] = 1;
  m_keys["RMS"] = 2;
  m_keys["M3"] = 3;
  m_keys["PNABMEAN"] = 4;
  m_keys["PNABRMS"] = 5;
  m_keys["PNABM3"] = 6;
  m_keys["FLAG"] = 7;
  m_keys["VMIN"] = 8;
  m_keys["VMAX"] = 9;

  m_type.resize(10);
  for (int i = 0; i < 10; i++) {
    m_type[i] = "NUMBER";
  }

  m_system = 0;
  m_color = 0;
}

bool LMFPnPrimDat::isValid() {
  bool ret = true;
  if ((getSystem() != "LASER") && (getSystem() != "LED")) {
    m_Error += " System name not properly set [" + getSystem() + "]";
    ret = false;
  }
  if ((getSystem() == "LASER") && (getColor() != "BLUE") && (getColor() != "IR")) {
    m_Error += " Color not properly set [" + getColor() + "]";
    ret = false;
  }
  if ((getSystem() == "LED") && (getColor() != "BLUE") && (getColor() != "ORANGE")) {
    m_Error += " Color not properly set [" + getColor() + "]";
    ret = false;
  }
  return ret;
}

LMFPnPrimDat &LMFPnPrimDat::setMean(EcalLogicID &id, float v) {
  LMFDat::setData(id, "MEAN", v);
  return *this;
}

LMFPnPrimDat &LMFPnPrimDat::setRMS(EcalLogicID &id, float v) {
  LMFDat::setData(id, "RMS", v);
  return *this;
}

LMFPnPrimDat &LMFPnPrimDat::setM3(EcalLogicID &id, float v) {
  LMFDat::setData(id, "M3", v);
  return *this;
}

LMFPnPrimDat &LMFPnPrimDat::setPN(EcalLogicID &id, float mean, float rms, float m3) {
  setMean(id, mean);
  setRMS(id, rms);
  setM3(id, m3);
  return *this;
}
LMFPnPrimDat &LMFPnPrimDat::setShapeCorr(EcalLogicID &id, float v) {
  if (getSystem() != "LED") {
    LMFDat::setData(id, "SHAPECORRPN", v);
  }
  return *this;
}

LMFPnPrimDat &LMFPnPrimDat::setPNAoverBM3(EcalLogicID &id, float v) {
  LMFDat::setData(id, "PNABM3", v);
  return *this;
}

LMFPnPrimDat &LMFPnPrimDat::setPNAoverBMean(EcalLogicID &id, float v) {
  LMFDat::setData(id, "PNABMEAN", v);
  return *this;
}

LMFPnPrimDat &LMFPnPrimDat::setPNAoverBRMS(EcalLogicID &id, float v) {
  LMFDat::setData(id, "PNABRMS", v);
  return *this;
}

LMFPnPrimDat &LMFPnPrimDat::setPNAoverB(EcalLogicID &id, float mean, float rms, float m3) {
  setPNAoverBMean(id, mean);
  setPNAoverBRMS(id, rms);
  setPNAoverBM3(id, m3);
  return *this;
}
LMFPnPrimDat &LMFPnPrimDat::setFlag(EcalLogicID &id, int v) {
  LMFDat::setData(id, "FLAG", v);
  return *this;
}

float LMFPnPrimDat::getMean(int id) { return getData(id, "MEAN"); }

float LMFPnPrimDat::getShapeCor(int id) {
  float x = 0;
  if (getSystem() != "LED") {
    x = getData(id, "SHAPECORRPN");
  }
  return x;
}

float LMFPnPrimDat::getRMS(int id) { return getData(id, "RMS"); }

float LMFPnPrimDat::getM3(int id) { return getData(id, "M3"); }

float LMFPnPrimDat::getPNAoverBM3(int id) { return getData(id, "PNABM3"); }

float LMFPnPrimDat::getPNAoverBMean(int id) { return getData(id, "PNABMEAN"); }

float LMFPnPrimDat::getPNAoverBRMS(int id) { return getData(id, "PNABRMS"); }

int LMFPnPrimDat::getFlag(int id) { return getData(id, "FLAG"); }

float LMFPnPrimDat::getMean(EcalLogicID &id) { return getData(id, "MEAN"); }

float LMFPnPrimDat::getShapeCor(EcalLogicID &id) {
  float x = 0.;
  if (getSystem() != "LED") {
    x = getData(id, "SHAPECORRPN");
  }
  return x;
}

float LMFPnPrimDat::getRMS(EcalLogicID &id) { return getData(id, "RMS"); }

float LMFPnPrimDat::getM3(EcalLogicID &id) { return getData(id, "M3"); }

float LMFPnPrimDat::getPNAoverBM3(EcalLogicID &id) { return getData(id, "PNABM3"); }

float LMFPnPrimDat::getPNAoverBMean(EcalLogicID &id) { return getData(id, "PNABMEAN"); }

float LMFPnPrimDat::getPNAoverBRMS(EcalLogicID &id) { return getData(id, "PNABRMS"); }

int LMFPnPrimDat::getFlag(EcalLogicID &id) { return getData(id, "FLAG"); }
