#include "DataFormats/TCDS/interface/L1aInfo.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"

L1aInfo::L1aInfo() : orbitNr_(0), bxid_(0), index_(0), eventType_(0) {}

L1aInfo::L1aInfo(const tcds::L1aInfo_v1& l1Info)
    : orbitNr_(((uint64_t)(l1Info.orbithigh) << 32) | l1Info.orbitlow),
      bxid_(l1Info.bxid),
      index_(-l1Info.ind0 - 1),
      eventType_(l1Info.eventtype) {}

std::ostream& operator<<(std::ostream& s, const L1aInfo& l1aInfo) {
  s << "Index:        " << l1aInfo.getIndex() << std::endl;
  s << "   OrbitNr:   " << l1aInfo.getOrbitNr() << std::endl;
  s << "   BXID:      " << l1aInfo.getBXID() << std::endl;
  s << "   EventType: " << (uint16_t)l1aInfo.getEventType() << std::endl;

  return s;
}
