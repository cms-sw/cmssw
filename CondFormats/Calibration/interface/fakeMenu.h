#ifndef CondFormats_fakeMenu_h
#define CondFormats_fakeMenu_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
class Algo {
public:
  Algo() {}
  int a;

  COND_SERIALIZABLE;
};
class AlgoMap : public std::map<std::string, Algo> {
public:
  AlgoMap() {}

  COND_SERIALIZABLE;
};
class fakeMenu {
public:
  // constructor
  fakeMenu() {}
  virtual ~fakeMenu() {}

private:
  AlgoMap m_algorithmMap;

  COND_SERIALIZABLE;
};
#endif
