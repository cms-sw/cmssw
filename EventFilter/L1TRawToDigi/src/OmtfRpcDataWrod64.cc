#include "EventFilter/L1TRawToDigi/interface/OmtfRpcDataWord64.h"

#include <bitset>

namespace omtf {
  std::ostream & operator<< (std::ostream &out, const RpcDataWord64 &o) {
    out << "RpcDataWord64: "
        <<" type: "<< DataWord64::type(o.type())
        << " bx: "<<o.bxNum_
        << " lnk: "<< o.linkNum_;
    out << std::hex;
    out << " frame1: 0x"<< o.frame1_; if (o.frame1_ != 0) out <<" ("<< std::bitset<16>(o.frame1_)<<")";
    out << " frame2: 0x"<< o.frame2_; if (o.frame2_ != 0) out <<" ("<< std::bitset<16>(o.frame2_)<<")";
    out << " frame3: 0x"<< o.frame3_; if (o.frame3_ != 0) out <<" ("<< std::bitset<16>(o.frame3_)<<")";
    out << std::dec;
    return out;
  }
}

