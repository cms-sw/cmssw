
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

#include <utility>

// Namespace resolution
using std::map;
using std::pair;
using std::string;

GctBlockHeader::GctBlockHeader(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid):
  GctBlockHeaderBase()
{
  d = (id & 0xff) + ((nsamples&0xf)<<8) + ((bcid&0xfff)<<12) + ((evid&0xff)<<24);
}

/// setup class static to lookup block length
pair<unsigned, unsigned> a[] = {
  pair<unsigned, unsigned>(0x00,0),
  pair<unsigned, unsigned>(0x58,6),
  pair<unsigned, unsigned>(0x59,12),
  pair<unsigned, unsigned>(0x5a,2),
  pair<unsigned, unsigned>(0x5f,4),
  pair<unsigned, unsigned>(0x68,4),
  pair<unsigned, unsigned>(0x69,16),
  pair<unsigned, unsigned>(0x6a,2),
  pair<unsigned, unsigned>(0x6b,2),
  pair<unsigned, unsigned>(0x6f,4),
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
  pair<unsigned, unsigned>(0xff,198)  // Our temp hack RCT calo block
};

GctBlockHeaderBase::BlockLengthMap GctBlockHeader::blockLength_(a, a + sizeof(a) / sizeof(a[0]));

/// setup class static to lookup block name
pair<unsigned, string> b[] = {
  pair<unsigned, string>(0x00,"NULL"),
  pair<unsigned, string>(0x58,"ConcJet: Jet Cands Output to Global Trigger"),
  pair<unsigned, string>(0x59,"ConcJet: Sort Input"),
  pair<unsigned, string>(0x5a,"ConcJet: Jet Counts Output to Global Trigger"),
  pair<unsigned, string>(0x5f,"ConcJet: Bunch Counter Pattern Test"),
  pair<unsigned, string>(0x68,"ConcElec: EM Cands Output to Global Trigger"),
  pair<unsigned, string>(0x69,"ConcElec: Sort Input"),
  pair<unsigned, string>(0x6a,"ConcElec: Energy Sums Output to Global Trigger"),
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
  pair<unsigned, string>(0xff,"All RCT Calo Regions")  // Our temp hack RCT calo block
};

GctBlockHeaderBase::BlockNameMap GctBlockHeader::blockName_(b, b + sizeof(b) / sizeof(b[0]));

