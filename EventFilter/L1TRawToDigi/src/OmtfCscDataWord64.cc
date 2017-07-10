#include "EventFilter/L1TRawToDigi/interface/OmtfCscDataWord64.h"

namespace omtf {
std::ostream & operator<< (std::ostream &out, const CscDataWord64 &o) {
    out << "CscDataWord64: "
        <<" type: "<< DataWord64::type(o.type())
        << " val: "<< o.valid()
        << " bx: "<<o.bxNum()
        << " lnk: "<< o.linkNum()
        << " stat: "<<o.station()
        << " cscId: " << o.cscID()
        << " hit: "<< o.hitNum()
        << " qual: "<< o.quality()
        << " patt: " << o.clctPattern()
        << " bending: " << o.bend()
        << " hs: "<<o.halfStrip()
        << " wg: "<< o.wireGroup();
    return out;
  }
}
