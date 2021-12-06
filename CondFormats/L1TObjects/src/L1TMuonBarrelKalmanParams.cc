#include "CondFormats/L1TObjects/interface/L1TMuonBarrelKalmanParams.h"

L1TMuonBarrelKalmanParams::L1TMuonBarrelKalmanParams() : pnodes_(NUM_BMTF_PARAM_NODES), l1mudttfmasks_(1) {
  version_ = Version;
  pnodes_[CONFIG].type_ = "unspecified";
  pnodes_[CONFIG].fwVersion_ = 0;  //default to recognize a RCD that is not filled
}
