
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

#include <utility>

using std::map;
using std::pair;
using std::string;

GctBlockHeader::GctBlockHeader(const uint32_t data) { d = data; }

GctBlockHeader::GctBlockHeader(const unsigned char * data) { 
  d = data[0] + (data[1]<<8) + (data[2]<<16) + (data[3]<<24);
}

GctBlockHeader::GctBlockHeader(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid) {
  d = (id & 0xff) + ((nsamples&0xf)<<8) + ((bcid&0xfff)<<12) + ((evid&0xff)<<24);
}

GctBlockHeader::~GctBlockHeader() { }

std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h) {
  os << "GCT Raw Data Block : " << h.name() << " : ID " << std::hex << h.id() << " : Length : " << h.length() << " : Samples " << h.nSamples() << " : BX " << h.bcId() << " : Event " << h.eventId() << std::dec;
  return os;
}


/// setup class static to lookup block length
pair<unsigned, unsigned> a[] = {
  pair<unsigned, unsigned>(0x00,0),
  pair<unsigned, unsigned>(0x58,0),
  pair<unsigned, unsigned>(0x59,0),
  pair<unsigned, unsigned>(0x5f,1),
  pair<unsigned, unsigned>(0x68,4),
  pair<unsigned, unsigned>(0x69,16),
  pair<unsigned, unsigned>(0x6b,2),
  pair<unsigned, unsigned>(0x6f,1),
  pair<unsigned, unsigned>(0x80,20),
  pair<unsigned, unsigned>(0x81,15),
  pair<unsigned, unsigned>(0x83,4),
  pair<unsigned, unsigned>(0x88,16),
  pair<unsigned, unsigned>(0x89,12),
  pair<unsigned, unsigned>(0x8b,4),
  pair<unsigned, unsigned>(0xc0,20),
  pair<unsigned, unsigned>(0xc1,15),
  pair<unsigned, unsigned>(0xc3,4),
  pair<unsigned, unsigned>(0xc8,16),
  pair<unsigned, unsigned>(0xc9,12),
  pair<unsigned, unsigned>(0xcb,4),
};

map<unsigned, unsigned> GctBlockHeader::blockLength_(a, a + sizeof(a) / sizeof(a[0]));

/// setup class static to lookup block name
pair<unsigned, string> b[] = {
  pair<unsigned, string>(0x00,"NULL"),
  pair<unsigned, string>(0x58,"Greg's random register"),
  pair<unsigned, string>(0x59,"Greg's other random register"),
  pair<unsigned, string>(0x5f,"ConcJet: Bunch Counter Pattern Test"),
  pair<unsigned, string>(0x68,"ConcElec: Output to Global Trigger"),
  pair<unsigned, string>(0x69,"ConcElec: Sort Input"),
  pair<unsigned, string>(0x6b,"ConcElec: GT Serdes Loopback"),
  pair<unsigned, string>(0x6f,"ConcElec: Bunch Counter Pattern Test"),
  pair<unsigned, string>(0x80,"Leaf-U1, Elec, NegEta, Sort Input"),
  pair<unsigned, string>(0x81,"Leaf-U1, Elec, NegEta, Raw Input"),
  pair<unsigned, string>(0x83,"Leaf-U1, Elec, NegEta, Sort Output"),
  pair<unsigned, string>(0x88,"Leaf-U2, Elec, NegEta, Sort Input"),
  pair<unsigned, string>(0x89,"Leaf-U2, Elec, NegEta, Raw Input"),
  pair<unsigned, string>(0x8b,"Leaf-U2, Elec, NegEta, Sort Output"),
  pair<unsigned, string>(0xc0,"Leaf-U1, Elec, PosEta, Sort Input"),
  pair<unsigned, string>(0xc1,"Leaf-U1, Elec, PosEta, Raw Input"),
  pair<unsigned, string>(0xc3,"Leaf-U1, Elec, PosEta, Sort Output"),
  pair<unsigned, string>(0xc8,"Leaf-U2, Elec, PosEta, Sort Input"),
  pair<unsigned, string>(0xc9,"Leaf-U2, Elec, PosEta, Raw Input"),
  pair<unsigned, string>(0xcb,"Leaf-U2, Elec, PosEta, Sort Output"),
};

map<unsigned, string> GctBlockHeader::blockName_(b, b + sizeof(b) / sizeof(b[0]));

