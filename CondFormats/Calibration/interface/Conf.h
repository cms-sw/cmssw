#ifndef CondEx_KeyedConf_H
#define CondEx_KeyedConf_H
/*
 * Examples of configurations identified by a key
 */

#include "CondFormats/Common/interface/BaseKeyed.h"
#include <string>

namespace condex {

  struct ConfI : public  BaseKeyed {
    ConfI(std::string k, int i) : key(k), v(i){}
    int v;
    std::string key; // just for test
  };

  struct ConfF : public  BaseKeyed {
    ConfI(std::string k, float i) : key(k), v(i){}
    float v;
    std::string key; // just for test
  };


}
