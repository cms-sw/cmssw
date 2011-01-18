#include "OnlineDB/EcalCondDB/interface/LMFColoredTable.h"

LMFColoredTable::LMFColoredTable() : LMFDat() {
  m_className = "LMFColoredTable";
  m_system = 0;
  m_color = 0;
  COLOR[0]  = "BLUE";
  COLOR[1]  = "IR";
  COLOR[2]  = "ORANGE";
  SYSTEM[0] = "LASER";
  SYSTEM[1] = "LED";
}

LMFColoredTable::LMFColoredTable(EcalDBConnection *c) : LMFDat(c) {
  m_className = "LMFColoredTable";
  m_system = 0;
  m_color = 0;
  COLOR[0]  = "BLUE";
  COLOR[1]  = "IR";
  COLOR[2]  = "ORANGE";
  SYSTEM[0] = "LASER";
  SYSTEM[1] = "LED";
}

LMFColoredTable::LMFColoredTable(oracle::occi::Environment* env,
				 oracle::occi::Connection* conn) : 
  LMFDat(env, conn) {
  m_className = "LMFColoredTable";
  m_system = 0;
  m_color = 0;
  COLOR[0]  = "BLUE";
  COLOR[1]  = "IR";
  COLOR[2]  = "ORANGE";
  SYSTEM[0] = "LASER";
  SYSTEM[1] = "LED";
}

std::string LMFColoredTable::getColor() {
  return COLOR[m_color];
}

std::string LMFColoredTable::getSystem() {
  return SYSTEM[m_system];
}

LMFColoredTable& LMFColoredTable::setColor(std::string color) {
  std::map<int, std::string>::const_iterator i = COLOR.begin();
  std::map<int, std::string>::const_iterator e = COLOR.end();
  bool loop = true;
  while ((loop) && (i != e)) {
    if (i->second == color) {
      loop = false;
      setColor(i->first);
    }
    i++;
  }
  return *this;
}

LMFColoredTable& LMFColoredTable::setSystem(std::string system) {
  std::map<int, std::string>::const_iterator i = SYSTEM.begin();
  std::map<int, std::string>::const_iterator e = SYSTEM.end();
  bool loop = true;
  while ((loop) && (i != e)) {
    if (i->second == system) {
      loop = false;
      setSystem(i->first);
    }
    i++;
  }
  return *this;
}

int LMFColoredTable::writeDB() 
  throw(std::runtime_error) {
  // check if the VMIN version has been properly set, otherwise 
  // change it to the default value
  std::map<int, std::vector<float> >::iterator i = m_data.begin();
  std::map<int, std::vector<float> >::iterator e = m_data.end();
  while (i != e) {
    int s = i->second.size();
    if (i->second[s - 2] == 0) { // VMIN cannot be NULL
      i->second[s - 2] = 1;
    }
    i++;
  }
  return LMFDat::writeDB();
}
