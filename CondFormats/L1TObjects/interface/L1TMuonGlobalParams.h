///
/// \class L1TMuonGlobalParams
///
/// Description: Placeholder for MicroGMT parameters
///
/// Implementation:
///
/// \author: Thomas Reis
///

#ifndef L1TMuonGlobalParams_h
#define L1TMuonGlobalParams_h

#include <memory>
#include <iostream>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/LUT.h"

class L1TMuonGlobalParams {

public:
  enum { Version = 1 };

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

  L1TMuonGlobalParams() : pnodes_(0) { version_=Version; }
  ~L1TMuonGlobalParams() {}

protected:
  unsigned version_;

  std::vector<Node> pnodes_;

  COND_SERIALIZABLE;
};
#endif
