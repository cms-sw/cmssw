#include "CondFormats/L1TObjects/interface/L1TwinMuxParams.h"

L1TwinMuxParams::L1TwinMuxParams() :
  pnodes_(NUM_TM_PARAM_NODES)
{
  //pnodes_[CONFIG].sparams_.clear();
  pnodes_[CONFIG].iparams_.resize(NUM_CONFIG_PARAMS);
  version_=Version;
}

void L1TwinMuxParams::print(std::ostream& out) const {

  out << "L1 TwinMux Parameters" << std::endl;

  out << "Firmware version: " << fwVersion_ << std::endl;
}
