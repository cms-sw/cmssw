
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

GctBlockHeader::GctBlockHeader(const uint32_t data) { d = data; }

GctBlockHeader::GctBlockHeader(const unsigned char * data) { 
  d = data[0] + (data[1]<<8) + (data[2]<<16) + (data[3]<<24);
}

GctBlockHeader::GctBlockHeader(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid) {
  d = (id & 0xff) + ((nsamples&0xf)<<8) + ((bcid&0xfff)<<12) + ((evid&0xff)<<24);
}

GctBlockHeader::~GctBlockHeader() { }

std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h) {
  os << "ID " << std::hex << h.id() << " : Samples " << h.nSamples() << " : BX " << h.bcId() << " : Event " << h.eventId() << std::dec;
  return os;
}
