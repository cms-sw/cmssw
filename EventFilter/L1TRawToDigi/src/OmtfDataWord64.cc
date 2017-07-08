#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"

std::ostream & Omtf::DataWord64::operator<< (std::ostream &out, const Omtf::DataWord64::Type &o) {
  switch(o) {
    case(csc)  : out <<"csc "; break;
    case(rpc)  : out <<"rpc "; break;
    case(dt)   : out <<"dt  "; break;
    case(omtf) : out <<"omtf"; break;
    default    : out <<"unkn"; break;
  }
  out<<"(0x"<<std::hex<<static_cast<int>(o)<<std::dec<<")";
  return out;
}
