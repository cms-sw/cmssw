#ifndef LMFCOLOREDTABLE_H
#define LMFCOLOREDTABLE_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFDat.h"
#include <map>

/**
 *   LMF_YYYY_XXX_ZZZ interface
 *        ^    ^   ^
 *        |    |   |
 *        |    |   \- table
 *        |    \_____ color
 *        \---------- system
 */

class LMFColoredTable : public LMFDat {
 public:
  LMFColoredTable();
  LMFColoredTable(EcalDBConnection *c);
  LMFColoredTable(oracle::occi::Environment* env,
	          oracle::occi::Connection* conn);
  ~LMFColoredTable() {}

  virtual std::string getTableName() const = 0;

  std::string getColor() const;
  virtual std::string getSystem() const;

  LMFColoredTable& setColor(int color) {
    if (COLOR.find(color) != COLOR.end()) {
      m_color = color;
      m_className += "/C=" + COLOR[color];
    }
    return *this;
  }
  LMFColoredTable& setColor(std::string color);
  virtual LMFColoredTable& setSystem(int system) {
    if (SYSTEM.find(system) != SYSTEM.end()) {
      m_system = system;
      m_className += "/S=" + SYSTEM[system];
    }
    return *this;
  }
  virtual LMFColoredTable& setSystem(std::string s);
  LMFColoredTable& setVmin(EcalLogicID &id, int v) {
    setData(id, "VMIN", v);
    return *this;
  }
  LMFColoredTable& setVmax(EcalLogicID &id, int v) {
    setData(id, "VMAX", v);
    return *this;
  }

  LMFColoredTable& setVersions(EcalLogicID &id, int vmin, int vmax) {
    setData(id, "VMIN", vmin);
    setData(id, "VMAX", vmax);
    return *this;
  }

  int getVmin(EcalLogicID &id) {
    return getData(id, "VMIN");
  }

  int getVmax(EcalLogicID &id) {
    return getData(id, "VMAX");
  }

  int writeDB() throw(std::runtime_error);


 protected:
  int m_color;
  int m_system;

 private:
  std::map<int, std::string> COLOR;
  std::map<int, std::string> SYSTEM;
};

#endif
