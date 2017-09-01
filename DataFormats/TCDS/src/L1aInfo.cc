#include "DataFormats/TCDS/interface/L1aInfo.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"

L1aInfo::L1aInfo() :
  orbit_(0),
  bxid_(0),
  eventType_(0)
{}


L1aInfo::L1aInfo(const tcds::L1aInfo_v1& l1Info) :
  orbit_(((uint64_t)(l1Info.orbithigh)<<32)|l1Info.orbitlow),
  bxid_(l1Info.bxid),
  eventType_(l1Info.eventtype)
{}
