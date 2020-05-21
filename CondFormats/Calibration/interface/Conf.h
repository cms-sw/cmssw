#ifndef CondEx_KeyedConf_H
#define CondEx_KeyedConf_H
/*
 * Examples of configurations identified by a key
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/Common/interface/BaseKeyed.h"
#include <string>
#include <utility>

namespace condex {

  struct ConfI : public cond::BaseKeyed {
    ConfI() : v(0), key(" ") {}
    ConfI(std::string k, int i) : v(i), key(std::move(k)) {}
    int v;
    std::string key;  // just for test

    COND_SERIALIZABLE;
  };

  struct ConfF : public cond::BaseKeyed {
    ConfF() : v(0), key(" ") {}
    ConfF(std::string k, float i) : v(i), key(std::move(k)) {}
    float v;
    std::string key;  // just for test

    COND_SERIALIZABLE;
  };

}  // namespace condex

#endif
