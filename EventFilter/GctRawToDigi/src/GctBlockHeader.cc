#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

// Namespace resolution
using std::string;

GctBlockHeader::GctBlockHeader(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid):
  GctBlockHeaderBase()
{
  d = (id & 0xff) + ((nsamples&0xf)<<8) + ((bcid&0xfff)<<12) + ((evid&0xff)<<24);
}

/// setup class static to lookup block length
GctBlockHeaderBase::BlockLengthPair initArray1[] = {
  GctBlockHeaderBase::BlockLengthPair(0x00,0),
  GctBlockHeaderBase::BlockLengthPair(0x58,6),
  GctBlockHeaderBase::BlockLengthPair(0x59,12),
  GctBlockHeaderBase::BlockLengthPair(0x5a,2),
  GctBlockHeaderBase::BlockLengthPair(0x5f,4),
  GctBlockHeaderBase::BlockLengthPair(0x68,4),
  GctBlockHeaderBase::BlockLengthPair(0x69,16),
  GctBlockHeaderBase::BlockLengthPair(0x6a,2),
  GctBlockHeaderBase::BlockLengthPair(0x6b,2),
  GctBlockHeaderBase::BlockLengthPair(0x6f,4),
  GctBlockHeaderBase::BlockLengthPair(0x80,20),
  GctBlockHeaderBase::BlockLengthPair(0x81,15),
  GctBlockHeaderBase::BlockLengthPair(0x83,4),
  GctBlockHeaderBase::BlockLengthPair(0x88,16),
  GctBlockHeaderBase::BlockLengthPair(0x89,12),
  GctBlockHeaderBase::BlockLengthPair(0x8b,4),
  GctBlockHeaderBase::BlockLengthPair(0xc0,20),
  GctBlockHeaderBase::BlockLengthPair(0xc1,15),
  GctBlockHeaderBase::BlockLengthPair(0xc3,4),
  GctBlockHeaderBase::BlockLengthPair(0xc8,16),
  GctBlockHeaderBase::BlockLengthPair(0xc9,12),
  GctBlockHeaderBase::BlockLengthPair(0xcb,4),
  GctBlockHeaderBase::BlockLengthPair(0xff,198)  // Our temp hack RCT calo block
};

GctBlockHeaderBase::BlockLengthMap GctBlockHeader::blockLength_(initArray1, initArray1 + sizeof(initArray1) / sizeof(initArray1[0]));

/// setup class static to lookup block name
GctBlockHeaderBase::BlockNamePair initArray2[] = {
  GctBlockHeaderBase::BlockNamePair(0x00,"NULL"),
  GctBlockHeaderBase::BlockNamePair(0x58,"ConcJet: Jet Cands Output to Global Trigger"),
  GctBlockHeaderBase::BlockNamePair(0x59,"ConcJet: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x5a,"ConcJet: Jet Counts Output to Global Trigger"),
  GctBlockHeaderBase::BlockNamePair(0x5f,"ConcJet: Bunch Counter Pattern Test"),
  GctBlockHeaderBase::BlockNamePair(0x68,"ConcElec: EM Cands Output to Global Trigger"),
  GctBlockHeaderBase::BlockNamePair(0x69,"ConcElec: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x6a,"ConcElec: Energy Sums Output to Global Trigger"),
  GctBlockHeaderBase::BlockNamePair(0x6b,"ConcElec: GT Serdes Loopback"),
  GctBlockHeaderBase::BlockNamePair(0x6f,"ConcElec: Bunch Counter Pattern Test"),
  GctBlockHeaderBase::BlockNamePair(0x80,"Leaf-U1, Elec, NegEta, Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x81,"Leaf-U1, Elec, NegEta, Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x83,"Leaf-U1, Elec, NegEta, Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0x88,"Leaf-U2, Elec, NegEta, Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x89,"Leaf-U2, Elec, NegEta, Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x8b,"Leaf-U2, Elec, NegEta, Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0xc0,"Leaf-U1, Elec, PosEta, Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0xc1,"Leaf-U1, Elec, PosEta, Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xc3,"Leaf-U1, Elec, PosEta, Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0xc8,"Leaf-U2, Elec, PosEta, Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0xc9,"Leaf-U2, Elec, PosEta, Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xcb,"Leaf-U2, Elec, PosEta, Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0xff,"All RCT Calo Regions")  // Our temp hack RCT calo block
};

GctBlockHeaderBase::BlockNameMap GctBlockHeader::blockName_(initArray2, initArray2 + sizeof(initArray2) / sizeof(initArray2[0]));

