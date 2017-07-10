#include "EventFilter/L1TRawToDigi/interface/OmtfDtDataWord64.h"

namespace omtf {
  std::ostream & operator<< (std::ostream &out, const DtDataWord64 &o) {
    out << "DtDataWord64:  "
        <<" type: "<< DataWord64::type(o.type())
        << " bx: "<<o.bxNum()
        << " station: " << o.station()
        << " sector: " << o.sector()
        << " fiber: " << o.fiber()
        << " phi: "<<o.phi()
        << " quality: "<<o.quality()
        << " eta: "<<o.eta()
        << " etaQ: "<<o.etaQuality()
        << " bcnt: "<<o.bcnt_st()<<"_"<<o.bcnt_e0()<<"_"<<o.bcnt_e1()
        << "";
    return out;
  }
}
