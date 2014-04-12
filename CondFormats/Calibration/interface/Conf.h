#ifndef CondEx_KeyedConf_H
#define CondEx_KeyedConf_H
/*
 * Examples of configurations identified by a key
 */

#include "CondFormats/Common/interface/BaseKeyed.h"
#include <string>

namespace condex {

  struct ConfI : public cond::BaseKeyed {
    ConfI()  : v(0), key(" ") {}
    ConfI(std::string k, int i) : v(i), key(k) {}
    int v;
    std::string key; // just for test
  };

  struct ConfF : public cond::BaseKeyed {
    ConfF() : v(0), key(" ") {}
    ConfF(std::string k, float i) : v(i), key(k) {}
    float v;
    std::string key; // just for test
  };

}

#endif
