
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


// /// setup class statics to lookup block length

// GctBlockHeader::blockLength_[0x00] = 0;
// GctBlockHeader::blockLength_[0x58] = 0;
// GctBlockHeader::blockLength_[0x59] = 0;
// GctBlockHeader::blockLength_[0x5f] = 1;   // ConcJet: Bunch Counter Pattern Test
// GctBlockHeader::blockLength_[0x68] = 4;   // ConcElec: Output to Global Trigger
// GctBlockHeader::blockLength_[0x69] = 16;  // ConcElec: Sort Input
// GctBlockHeader::blockLength_[0x6b] = 2;   // ConcElec: GT Serdes Loopback
// GctBlockHeader::blockLength_[0x6f] = 1;   // ConcElec: Bunch Counter Pattern Test
// GctBlockHeader::blockLength_[0x80] = 20;  // Leaf-U1, Elec, NegEta, Sort Input
// GctBlockHeader::blockLength_[0x81] = 15;  // Leaf-U1, Elec, NegEta, Raw Input
// GctBlockHeader::blockLength_[0x83] = 4;   // Leaf-U1, Elec, NegEta, Sort Output
// GctBlockHeader::blockLength_[0x88] = 16;  // Leaf-U2, Elec, NegEta, Sort Input
// GctBlockHeader::blockLength_[0x89] = 12;  // Leaf-U2, Elec, NegEta, Raw Input
// GctBlockHeader::blockLength_[0x8b] = 4;   // Leaf-U2, Elec, NegEta, Sort Output
// GctBlockHeader::blockLength_[0xc0] = 20;  // Leaf-U1, Elec, PosEta, Sort Input
// GctBlockHeader::blockLength_[0xc1] = 15;  // Leaf-U1, Elec, PosEta, Raw Input
// GctBlockHeader::blockLength_[0xc3] = 4;   // Leaf-U1, Elec, PosEta, Sort Output
// GctBlockHeader::blockLength_[0xc8] = 16;  // Leaf-U2, Elec, PosEta, Sort Input
// GctBlockHeader::blockLength_[0xc9] = 12;  // Leaf-U2, Elec, PosEta, Raw Input
// GctBlockHeader::blockLength_[0xcb] = 4;   // Leaf-U2, Elec, PosEta, Sort Output

// // setup block names
// GctBlockHeader::blockName_[0x00] = "null block";
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x59] = ;
// GctBlockHeader::blockName_[0x5f] = ;
// GctBlockHeader::blockName_[0x68] = ;
// GctBlockHeader::blockName_[0x69] = ;
// GctBlockHeader::blockName_[0x6b] = ;
// GctBlockHeader::blockName_[0x6f] = ;
// GctBlockHeader::blockName_[0x80] = ;
// GctBlockHeader::blockName_[0x81] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
// GctBlockHeader::blockName_[0x58] = ;
