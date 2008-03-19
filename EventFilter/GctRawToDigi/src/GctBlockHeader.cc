#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

// Namespace resolution
using std::string;

GctBlockHeader::GctBlockHeader(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid):
  GctBlockHeaderBase()
{
  d = (id & 0xff) + ((nsamples&0xf)<<8) + ((bcid&0xfff)<<12) + ((evid&0xff)<<24);
}

/// setup class static to lookup block length
GctBlockHeader::BlockLengthPair initArray1[] = {
  GctBlockHeader::BlockLengthPair(0x00,0),
  GctBlockHeader::BlockLengthPair(0x58,6),
  GctBlockHeader::BlockLengthPair(0x59,12),
  GctBlockHeader::BlockLengthPair(0x5a,2),
  GctBlockHeader::BlockLengthPair(0x5f,4),
  GctBlockHeader::BlockLengthPair(0x68,4),
  GctBlockHeader::BlockLengthPair(0x69,16),
  GctBlockHeader::BlockLengthPair(0x6a,2),
  GctBlockHeader::BlockLengthPair(0x6b,2),
  GctBlockHeader::BlockLengthPair(0x6f,4),
  GctBlockHeader::BlockLengthPair(0x80,20),
  GctBlockHeader::BlockLengthPair(0x81,15),
  GctBlockHeader::BlockLengthPair(0x83,4),
  GctBlockHeader::BlockLengthPair(0x88,16),
  GctBlockHeader::BlockLengthPair(0x89,12),
  GctBlockHeader::BlockLengthPair(0x8b,4),
  GctBlockHeader::BlockLengthPair(0xc0,20),
  GctBlockHeader::BlockLengthPair(0xc1,15),
  GctBlockHeader::BlockLengthPair(0xc3,4),
  GctBlockHeader::BlockLengthPair(0xc8,16),
  GctBlockHeader::BlockLengthPair(0xc9,12),
  GctBlockHeader::BlockLengthPair(0xcb,4),
  GctBlockHeader::BlockLengthPair(0xff,198)  // Our temp hack RCT calo block
};

GctBlockHeader::BlockLengthMap GctBlockHeader::blockLength_(initArray1, initArray1 + sizeof(initArray1) / sizeof(initArray1[0]));

/// setup class static to lookup block name
GctBlockHeader::BlockNamePair initArray2[] = {
  GctBlockHeader::BlockNamePair(0x00,"NULL"),
  GctBlockHeader::BlockNamePair(0x58,"ConcJet: Jet Cands Output to Global Trigger"),
  GctBlockHeader::BlockNamePair(0x59,"ConcJet: Sort Input"),
  GctBlockHeader::BlockNamePair(0x5a,"ConcJet: Jet Counts Output to Global Trigger"),
  GctBlockHeader::BlockNamePair(0x5f,"ConcJet: Bunch Counter Pattern Test"),
  GctBlockHeader::BlockNamePair(0x68,"ConcElec: EM Cands Output to Global Trigger"),
  GctBlockHeader::BlockNamePair(0x69,"ConcElec: Sort Input"),
  GctBlockHeader::BlockNamePair(0x6a,"ConcElec: Energy Sums Output to Global Trigger"),
  GctBlockHeader::BlockNamePair(0x6b,"ConcElec: GT Serdes Loopback"),
  GctBlockHeader::BlockNamePair(0x6f,"ConcElec: Bunch Counter Pattern Test"),
  GctBlockHeader::BlockNamePair(0x80,"Leaf-U1, Elec, NegEta, Sort Input"),
  GctBlockHeader::BlockNamePair(0x81,"Leaf-U1, Elec, NegEta, Raw Input"),
  GctBlockHeader::BlockNamePair(0x83,"Leaf-U1, Elec, NegEta, Sort Output"),
  GctBlockHeader::BlockNamePair(0x88,"Leaf-U2, Elec, NegEta, Sort Input"),
  GctBlockHeader::BlockNamePair(0x89,"Leaf-U2, Elec, NegEta, Raw Input"),
  GctBlockHeader::BlockNamePair(0x8b,"Leaf-U2, Elec, NegEta, Sort Output"),
  GctBlockHeader::BlockNamePair(0xc0,"Leaf-U1, Elec, PosEta, Sort Input"),
  GctBlockHeader::BlockNamePair(0xc1,"Leaf-U1, Elec, PosEta, Raw Input"),
  GctBlockHeader::BlockNamePair(0xc3,"Leaf-U1, Elec, PosEta, Sort Output"),
  GctBlockHeader::BlockNamePair(0xc8,"Leaf-U2, Elec, PosEta, Sort Input"),
  GctBlockHeader::BlockNamePair(0xc9,"Leaf-U2, Elec, PosEta, Raw Input"),
  GctBlockHeader::BlockNamePair(0xcb,"Leaf-U2, Elec, PosEta, Sort Output"),
  GctBlockHeader::BlockNamePair(0xff,"All RCT Calo Regions")  // Our temp hack RCT calo block
};

GctBlockHeader::BlockNameMap GctBlockHeader::blockName_(initArray2, initArray2 + sizeof(initArray2) / sizeof(initArray2[0]));

