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
    /* std::vector<double> dparams_; */
    /* std::vector<unsigned> uparams_; */
    /* std::vector<int> iparams_; */
    /* std::vector<std::string> sparams_; */
    COND_SERIALIZABLE;
  };

  //enum {
  //  initialK,
  //  initialK2,
  //  eLoss,
  //  aPhi,
  //  aPhiB,
  //  aPhiBNLO,
  //  bPhi,
  //  bPhiB,
  //  phiAt2,
  //  etaLUT0,
  //  etaLUT1,
  //  //generic cuts
  //  chiSquare,
  //  chiSquareCutPattern,
  //  chiSquareCutCurvMax,
  //  chiSquareCut,
  //  //vertex cuts
  //  trackComp,
  //  trackCompErr1,
  //  trackCompErr2,
  //  trackCompCutPattern,
  //  trackCompCutCurvMax,
  //  trackCompCut,
  //  chiSquareCutTight,
  //  //combos
  //  combos4,
  //  combos3,
  //  combos2,
  //  combos1,
  //  //
  //  useOfflineAlgo,
  //  // Only for the offline algo -not in firmware -------------------- (possible use in phase2 ???)
  //  mScatteringPhi,
  //  mScatteringPhiB,
  //  pointResolutionPhi,
  //  pointResolutionPhiB,
  //  pointResolutionPhiBH,
  //  pointResolutionPhiBL,
  //  pointResolutionVertex,
  //  //
  //  NUM_CONFIG_PARAMS //to check
  //};

  // THIS SECTION IS TO BE USED FOR HANDLING THE MASKS
  // after initial integration with downstream code, a small update will change:
  //L1MuDTTFMasks l1mudttfmasks;
  // to this:
  //L1MuDTTFMasks &      l1mudttfmasks(){return l1mudttfmasks_[0]; }

  unsigned version_;
  std::vector<Node> pnodes_;

  COND_SERIALIZABLE;
};
#endif
