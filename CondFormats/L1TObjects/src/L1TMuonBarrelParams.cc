#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

L1TMuonBarrelParams::L1TMuonBarrelParams() :
  pnodes_(NUM_BMTF_PARAM_NODES),  
  l1mudttfparams_(1),
  l1mudttfmasks_(1)
{
  pnodes_[CONFIG].sparams_.clear();
  pnodes_[CONFIG].iparams_.resize(NUM_CONFIG_PARAMS);
  version_=Version;
}

void L1TMuonBarrelParams::print(std::ostream& out) const {

  out << "L1 BMTF Parameters" << std::endl;

  out << "Firmware version: " << fwVersion_ << std::endl;
}
