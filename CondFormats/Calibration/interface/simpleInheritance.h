#ifndef CondFormats_simpleInheritance_h
#define CondFormats_simpleInheritance_h

#include "CondFormats/Serialization/interface/Serializable.h"
class mybase {
public:
  mybase() {}

  COND_SERIALIZABLE;
};
class child : public mybase {
public:
  child() {}
  int b;

  COND_SERIALIZABLE;
};
#endif
