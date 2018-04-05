#include "CondFormats/L1TObjects/interface/L1TTwinMuxParams.h"

L1TTwinMuxParams::L1TTwinMuxParams() :
  pnodes_(NUM_TM_PARAM_NODES)
{
  //pnodes_[CONFIG].sparams_.clear();
  pnodes_[CONFIG].iparams_.resize(NUM_CONFIG_PARAMS);
  version_=Version;
}

void L1TTwinMuxParams::print(std::ostream& out) const {

  out << "L1 TwinMux Parameters" << std::endl;

  out << "Firmware version: " << fwVersion_ << std::endl;
}
