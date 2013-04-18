#include "OnlineDB/EcalCondDB/interface/LMFColoredTable.h"

LMFColoredTable::LMFColoredTable() : LMFDat() {
  m_className = "LMFColoredTable";
  m_system = 0;
  m_color = 0;
  COLOR[0]  = "BLUE";
  COLOR[1]  = "GREEN";
  COLOR[2]  = "ORANGE";
  COLOR[3]  = "IR";
  SYSTEM[0] = "LASER";
  SYSTEM[1] = "LED";
}

LMFColoredTable::LMFColoredTable(EcalDBConnection *c) : LMFDat(c) {
  m_className = "LMFColoredTable";
  m_system = 0;
  m_color = 0;
  COLOR[0]  = "BLUE";
  COLOR[1]  = "GREEN";
  COLOR[2]  = "ORANGE";
  COLOR[3]  = "IR";
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
  COLOR[1]  = "GREEN";
  COLOR[2]  = "ORANGE";
  COLOR[3]  = "IR";
  SYSTEM[0] = "LASER";
  SYSTEM[1] = "LED";
}

std::string LMFColoredTable::getColor() const {
  std::string ret = "";
  std::map<int, std::string>::const_iterator i = COLOR.find(m_color);
  if (i != COLOR.end()) {
    ret = i->second;
  }
  return ret;
}

std::string LMFColoredTable::getSystem() const {
  std::string ret = "";
  std::map<int, std::string>::const_iterator i = SYSTEM.find(m_system);
  if (i != SYSTEM.end()) {
    ret = i->second;
  }
  return ret;
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
  std::list<int> versions; // the list of different versions
  while (i != e) {
    int s = i->second.size();
    if (i->second[s - 2] == 0) { // VMIN cannot be NULL
      i->second[s - 2] = 1;
    }
    versions.push_back(i->second[s - 1]);
    versions.push_back(i->second[s - 2]);
    versions.unique();
    i++;
  }
  //  checkVesrions(versions); // not yet used, in fact...
  int ret = 0;
  try {
    ret = LMFDat::writeDB();
  }
  catch (std::runtime_error &e) {
    m_conn->rollback();
    throw(e);
  }
  return ret;
}
