#include "EventFilter/L1TRawToDigi/interface/OmtfMuonDataWord64.h"

#include <bitset>

namespace omtf {
  std::ostream & operator<< (std::ostream &out, const MuonDataWord64 &o) {
    out << "MuonDataWord64: "
        <<" type: "<< DataWord64::type(o.type())
        << " bx: "<<o.bxNum()
        << " pT: "<<o.pT()
        << " eta: "<<o.eta()
        << " phi: "<<o.phi()
        << " quality: "<<o.quality()
        << " layers: "<< std::bitset<18>(o.layers())
        << "";
    return out;
  }
}
