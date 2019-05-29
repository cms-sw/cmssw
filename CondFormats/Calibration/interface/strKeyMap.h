#ifndef CondFormats_strKeyMap_h
#define CondFormats_strKeyMap_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
class Algob {
public:
  Algob() {}
  int b;

  COND_SERIALIZABLE;
};
class strKeyMap {
public:
  strKeyMap() {}

private:
  std::map<std::string, Algob> m_content;

  COND_SERIALIZABLE;
};
#endif
