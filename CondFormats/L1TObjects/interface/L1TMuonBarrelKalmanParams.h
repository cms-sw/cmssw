///
/// \class L1TMuonBarrelKalmanParams
///
/// Description: Placeholder for Kalman BMTF parameters
///
///
/// \author: Panos Katsoulis
///

#ifndef L1TBMTFKalmanParams_h
#define L1TBMTFKalmanParams_h

#include <memory>
#include <iostream>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/L1TObjects/interface/LUT.h"

class L1TMuonBarrelKalmanParams {
public:
  L1TMuonBarrelKalmanParams();
  ~L1TMuonBarrelKalmanParams() {}

  enum { Version = 1 };
  enum { CONFIG = 0, NUM_BMTF_PARAM_NODES = 2 };

  class Node {
  public:
    std::string type_;
    std::string kalmanLUTsPath_;
    unsigned fwVersion_;
    l1t::LUT LUT_;
    COND_SERIALIZABLE;
  };

  L1MuDTTFMasks l1mudttfmasks;
  unsigned version_;

  std::vector<Node> pnodes_;
  std::vector<L1MuDTTFMasks> l1mudttfmasks_;
  COND_SERIALIZABLE;
};
#endif
