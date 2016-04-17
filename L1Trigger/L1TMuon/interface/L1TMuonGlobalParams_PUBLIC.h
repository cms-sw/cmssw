//
// \class L1TMuonGlobalParams_PUBLIC
//
// We are delegating the interpretation of our CondFormats to helper classes.
//
// To do so, we need to make the persistent data public (or add a friend class) but that will require
// ALCA/DB signoff...  while we wait for that, we have this measure, which effectively casts away the private.
//
// This will go away once ALCA/DB signs off on our CondFormat clean up.

#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"

#ifndef L1TMuonGlobalParams_PUBLIC_h
#define L1TMuonGlobalParams_PUBLIC_h

#include <memory>
#include <iostream>
#include <vector>
#include <cassert>

class L1TMuonGlobalParams_PUBLIC {
public:
  class Node {
  public:
    std::string type_;
    unsigned version_;
    l1t::LUT LUT_;
    std::vector<double> dparams_;
    std::vector<unsigned> uparams_;
    std::vector<int> iparams_;
    std::vector<std::string> sparams_;
    Node(){ type_="unspecified"; version_=0; }
    COND_SERIALIZABLE;
  };
  unsigned version_;
  unsigned fwVersion_; //obsolete
  
  int bxMin_;  //obsolete
  int bxMax_;  //obsolete
  std::vector<Node> pnodes_;
};

const L1TMuonGlobalParams_PUBLIC & cast_to_L1TMuonGlobalParams_PUBLIC(const L1TMuonGlobalParams & x);

const L1TMuonGlobalParams & cast_to_L1TMuonGlobalParams(const L1TMuonGlobalParams_PUBLIC & x);

#endif
