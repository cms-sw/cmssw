
#include "EventFilter/GctRawToDigi/src/GctBlockHeaderV2.h"

#include <utility>

using std::map;
using std::pair;
using std::string;

GctBlockHeader::GctBlockHeader(const uint32_t data) { d = data; }

GctBlockHeader::GctBlockHeader(const unsigned char * data) {
  d = data[0] + (data[1]<<8) + (data[2]<<16) + (data[3]<<24);
}

GctBlockHeader::GctBlockHeader(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid) {
  d = (id & 0xfff) + ((nsamples&0xf)<<16) + ((bcid&0xfff)<<20) + ((evid&0xf)<<12);
}

GctBlockHeader::~GctBlockHeader() { }

unsigned int GctBlockHeader::length() const
{
  if(!valid()) { return 0; }
  return blockLength_[this->id()];
}

std::string GctBlockHeader::name() const
{
  if(!valid()) { return "Unknown/invalid block header"; }
  return blockName_[this->id()];
}

std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h) {
  os << "GCT Raw Data Block : " << h.name() << " : ID " << std::hex << h.id() << " : Length : " << h.length() << " : Samples " << h.nSamples() << " : BX " << h.bcId() << " : Event " << h.eventId() << std::dec;
  return os;
}


/// setup class static to lookup block length
pair<unsigned, unsigned> a[] = {
  pair<unsigned, unsigned>(0x000,0),
  pair<unsigned, unsigned>(0x583,8),
  pair<unsigned, unsigned>(0x580,12),
  pair<unsigned, unsigned>(0x587,4),
  pair<unsigned, unsigned>(0x683,6),
  pair<unsigned, unsigned>(0x680,16),
  pair<unsigned, unsigned>(0x686,2),
  pair<unsigned, unsigned>(0x687,4),
  pair<unsigned, unsigned>(0x800,20),
  pair<unsigned, unsigned>(0x804,15),
  pair<unsigned, unsigned>(0x803,4),
  pair<unsigned, unsigned>(0x880,16),
  pair<unsigned, unsigned>(0x884,12),
  pair<unsigned, unsigned>(0x883,4),
  pair<unsigned, unsigned>(0xc00,20),
  pair<unsigned, unsigned>(0xc04,15),
  pair<unsigned, unsigned>(0xc03,4),
  pair<unsigned, unsigned>(0xc80,16),
  pair<unsigned, unsigned>(0xc84,12),
  pair<unsigned, unsigned>(0xc83,4),      // -- end of strictly defined blocks --
  pair<unsigned, unsigned>(0x900,12),     // Leaf1JetPosEtaU1: JF2 Input
  pair<unsigned, unsigned>(0x904,8),      // Leaf1JetPosEtaU1: JF2 Raw Input
  pair<unsigned, unsigned>(0x901,3),      // Leaf1JetPosEtaU1: JF2 Shared Recieved
  pair<unsigned, unsigned>(0x902,3),      // Leaf1JetPosEtaU1: JF2 Shared Sent
  pair<unsigned, unsigned>(0x903,10),     // Leaf1JetPosEtaU1: JF2 Output
  pair<unsigned, unsigned>(0x908,12),     // Leaf1JetPosEtaU1: JF3 Input
  pair<unsigned, unsigned>(0x90c,8),      // Leaf1JetPosEtaU1: JF3 Raw Input
  pair<unsigned, unsigned>(0x909,3),      // Leaf1JetPosEtaU1: JF3 Shared Recieved
  pair<unsigned, unsigned>(0x90a,3),      // Leaf1JetPosEtaU1: JF3 Shared Sent
  pair<unsigned, unsigned>(0x90b,10),     // Leaf1JetPosEtaU1: JF3 Output
  pair<unsigned, unsigned>(0x980,3),      // Leaf1JetPosEtaU2: Eta0 Input
  pair<unsigned, unsigned>(0x984,6),      // Leaf1JetPosEtaU2: Eta0 Raw Input
  pair<unsigned, unsigned>(0x988,12),     // Leaf1JetPosEtaU2: JF1 Input
  pair<unsigned, unsigned>(0x98c,8),      // Leaf1JetPosEtaU2: JF1 Raw Input
  pair<unsigned, unsigned>(0x989,3),      // Leaf1JetPosEtaU2: JF1 Shared Recieved
  pair<unsigned, unsigned>(0x98a,3),      // Leaf1JetPosEtaU2: JF1 Shared Sent
  pair<unsigned, unsigned>(0x98b,10),     // Leaf1JetPosEtaU2: JF1 Output
  pair<unsigned, unsigned>(0xa00,12),     // Leaf2JetPosEtaU1: JF2 Input
  pair<unsigned, unsigned>(0xa04,8),      // Leaf2JetPosEtaU1: JF2 Raw Input
  pair<unsigned, unsigned>(0xa01,3),      // Leaf2JetPosEtaU1: JF2 Shared Recieved
  pair<unsigned, unsigned>(0xa02,3),      // Leaf2JetPosEtaU1: JF2 Shared Sent
  pair<unsigned, unsigned>(0xa03,10),     // Leaf2JetPosEtaU1: JF2 Output
  pair<unsigned, unsigned>(0xa08,12),     // Leaf2JetPosEtaU1: JF3 Input
  pair<unsigned, unsigned>(0xa0c,8),      // Leaf2JetPosEtaU1: JF3 Raw Input
  pair<unsigned, unsigned>(0xa09,3),      // Leaf2JetPosEtaU1: JF3 Shared Recieved
  pair<unsigned, unsigned>(0xa0a,3),      // Leaf2JetPosEtaU1: JF3 Shared Sent
  pair<unsigned, unsigned>(0xa0b,10),     // Leaf2JetPosEtaU1: JF3 Output
  pair<unsigned, unsigned>(0xa80,3),      // Leaf2JetPosEtaU2: Eta0 Input
  pair<unsigned, unsigned>(0xa84,6),      // Leaf2JetPosEtaU2: Eta0 Raw Input
  pair<unsigned, unsigned>(0xa88,12),     // Leaf2JetPosEtaU2: JF1 Input
  pair<unsigned, unsigned>(0xa8c,8),      // Leaf2JetPosEtaU2: JF1 Raw Input
  pair<unsigned, unsigned>(0xa89,3),      // Leaf2JetPosEtaU2: JF1 Shared Recieved
  pair<unsigned, unsigned>(0xa8a,3),      // Leaf2JetPosEtaU2: JF1 Shared Sent
  pair<unsigned, unsigned>(0xa8b,10),     // Leaf2JetPosEtaU2: JF1 Output
  pair<unsigned, unsigned>(0xb00,12),     // Leaf3JetPosEtaU1: JF2 Input
  pair<unsigned, unsigned>(0xb04,8),      // Leaf3JetPosEtaU1: JF2 Raw Input
  pair<unsigned, unsigned>(0xb01,3),      // Leaf3JetPosEtaU1: JF2 Shared Recieved
  pair<unsigned, unsigned>(0xb02,3),      // Leaf3JetPosEtaU1: JF2 Shared Sent
  pair<unsigned, unsigned>(0xb03,10),     // Leaf3JetPosEtaU1: JF2 Output
  pair<unsigned, unsigned>(0xb08,12),     // Leaf3JetPosEtaU1: JF3 Input
  pair<unsigned, unsigned>(0xb0c,8),      // Leaf3JetPosEtaU1: JF3 Raw Input
  pair<unsigned, unsigned>(0xb09,3),      // Leaf3JetPosEtaU1: JF3 Shared Recieved
  pair<unsigned, unsigned>(0xb0a,3),      // Leaf3JetPosEtaU1: JF3 Shared Sent
  pair<unsigned, unsigned>(0xb0b,10),     // Leaf3JetPosEtaU1: JF3 Output
  pair<unsigned, unsigned>(0xb80,3),      // Leaf3JetPosEtaU2: Eta0 Input
  pair<unsigned, unsigned>(0xb84,6),      // Leaf3JetPosEtaU2: Eta0 Raw Input
  pair<unsigned, unsigned>(0xb88,12),     // Leaf3JetPosEtaU2: JF1 Input
  pair<unsigned, unsigned>(0xb8c,8),      // Leaf3JetPosEtaU2: JF1 Raw Input
  pair<unsigned, unsigned>(0xb89,3),      // Leaf3JetPosEtaU2: JF1 Shared Recieved
  pair<unsigned, unsigned>(0xb8a,3),      // Leaf3JetPosEtaU2: JF1 Shared Sent
  pair<unsigned, unsigned>(0xb8b,10),     // Leaf3JetPosEtaU2: JF1 Output
  pair<unsigned, unsigned>(0xd00,12),     // Leaf1JetNegEtaU1: JF2 Input
  pair<unsigned, unsigned>(0xd04,8),      // Leaf1JetNegEtaU1: JF2 Raw Input
  pair<unsigned, unsigned>(0xd01,3),      // Leaf1JetNegEtaU1: JF2 Shared Recieved
  pair<unsigned, unsigned>(0xd02,3),      // Leaf1JetNegEtaU1: JF2 Shared Sent
  pair<unsigned, unsigned>(0xd03,10),     // Leaf1JetNegEtaU1: JF2 Output
  pair<unsigned, unsigned>(0xd08,12),     // Leaf1JetNegEtaU1: JF3 Input
  pair<unsigned, unsigned>(0xd0c,8),      // Leaf1JetNegEtaU1: JF3 Raw Input
  pair<unsigned, unsigned>(0xd09,3),      // Leaf1JetNegEtaU1: JF3 Shared Recieved
  pair<unsigned, unsigned>(0xd0a,3),      // Leaf1JetNegEtaU1: JF3 Shared Sent
  pair<unsigned, unsigned>(0xd0b,10),     // Leaf1JetNegEtaU1: JF3 Output
  pair<unsigned, unsigned>(0xd80,3),      // Leaf1JetNegEtaU2: Eta0 Input
  pair<unsigned, unsigned>(0xd84,6),      // Leaf1JetNegEtaU2: Eta0 Raw Input
  pair<unsigned, unsigned>(0xd88,12),     // Leaf1JetNegEtaU2: JF1 Input
  pair<unsigned, unsigned>(0xd8c,8),      // Leaf1JetNegEtaU2: JF1 Raw Input
  pair<unsigned, unsigned>(0xd89,3),      // Leaf1JetNegEtaU2: JF1 Shared Recieved
  pair<unsigned, unsigned>(0xd8a,3),      // Leaf1JetNegEtaU2: JF1 Shared Sent
  pair<unsigned, unsigned>(0xd8b,10),     // Leaf1JetNegEtaU2: JF1 Output
  pair<unsigned, unsigned>(0xe00,12),     // Leaf2JetNegEtaU1: JF2 Input
  pair<unsigned, unsigned>(0xe04,8),      // Leaf2JetNegEtaU1: JF2 Raw Input
  pair<unsigned, unsigned>(0xe01,3),      // Leaf2JetNegEtaU1: JF2 Shared Recieved
  pair<unsigned, unsigned>(0xe02,3),      // Leaf2JetNegEtaU1: JF2 Shared Sent
  pair<unsigned, unsigned>(0xe03,10),     // Leaf2JetNegEtaU1: JF2 Output
  pair<unsigned, unsigned>(0xe08,12),     // Leaf2JetNegEtaU1: JF3 Input
  pair<unsigned, unsigned>(0xe0c,8),      // Leaf2JetNegEtaU1: JF3 Raw Input
  pair<unsigned, unsigned>(0xe09,3),      // Leaf2JetNegEtaU1: JF3 Shared Recieved
  pair<unsigned, unsigned>(0xe0a,3),      // Leaf2JetNegEtaU1: JF3 Shared Sent
  pair<unsigned, unsigned>(0xe0b,10),     // Leaf2JetNegEtaU1: JF3 Output
  pair<unsigned, unsigned>(0xe80,3),      // Leaf2JetNegEtaU2: Eta0 Input
  pair<unsigned, unsigned>(0xe84,6),      // Leaf2JetNegEtaU2: Eta0 Raw Input
  pair<unsigned, unsigned>(0xe88,12),     // Leaf2JetNegEtaU2: JF1 Input
  pair<unsigned, unsigned>(0xe8c,8),      // Leaf2JetNegEtaU2: JF1 Raw Input
  pair<unsigned, unsigned>(0xe89,3),      // Leaf2JetNegEtaU2: JF1 Shared Recieved
  pair<unsigned, unsigned>(0xe8a,3),      // Leaf2JetNegEtaU2: JF1 Shared Sent
  pair<unsigned, unsigned>(0xe8b,10),     // Leaf2JetNegEtaU2: JF1 Output
  pair<unsigned, unsigned>(0xf00,12),     // Leaf3JetNegEtaU1: JF2 Input
  pair<unsigned, unsigned>(0xf04,8),      // Leaf3JetNegEtaU1: JF2 Raw Input
  pair<unsigned, unsigned>(0xf01,3),      // Leaf3JetNegEtaU1: JF2 Shared Recieved
  pair<unsigned, unsigned>(0xf02,3),      // Leaf3JetNegEtaU1: JF2 Shared Sent
  pair<unsigned, unsigned>(0xf03,10),     // Leaf3JetNegEtaU1: JF2 Output
  pair<unsigned, unsigned>(0xf08,12),     // Leaf3JetNegEtaU1: JF3 Input
  pair<unsigned, unsigned>(0xf0c,8),      // Leaf3JetNegEtaU1: JF3 Raw Input
  pair<unsigned, unsigned>(0xf09,3),      // Leaf3JetNegEtaU1: JF3 Shared Recieved
  pair<unsigned, unsigned>(0xf0a,3),      // Leaf3JetNegEtaU1: JF3 Shared Sent
  pair<unsigned, unsigned>(0xf0b,10),     // Leaf3JetNegEtaU1: JF3 Output
  pair<unsigned, unsigned>(0xf80,3),      // Leaf3JetNegEtaU2: Eta0 Input
  pair<unsigned, unsigned>(0xf84,6),      // Leaf3JetNegEtaU2: Eta0 Raw Input
  pair<unsigned, unsigned>(0xf88,12),     // Leaf3JetNegEtaU2: JF1 Input
  pair<unsigned, unsigned>(0xf8c,8),      // Leaf3JetNegEtaU2: JF1 Raw Input
  pair<unsigned, unsigned>(0xf89,3),      // Leaf3JetNegEtaU2: JF1 Shared Recieved
  pair<unsigned, unsigned>(0xf8a,3),      // Leaf3JetNegEtaU2: JF1 Shared Sent
  pair<unsigned, unsigned>(0xf8b,10)      // Leaf3JetNegEtaU2: JF1 Output
//  pair<unsigned, unsigned>(0x300,),       // START OF WHEEL FPGAS
//  pair<unsigned, unsigned>(0x303,),
//  pair<unsigned, unsigned>(0x380,),
//  pair<unsigned, unsigned>(0x383,),
//  pair<unsigned, unsigned>(0x700,),
//  pair<unsigned, unsigned>(0x703,),
//  pair<unsigned, unsigned>(0x780,),
//  pair<unsigned, unsigned>(0x783,),
};

map<unsigned, unsigned> GctBlockHeader::blockLength_(a, a + sizeof(a) / sizeof(a[0]));

/// setup class static to lookup block name
pair<unsigned, string> b[] = {
  pair<unsigned, string>(0x000,"NULL"),
  pair<unsigned, string>(0x583,"ConcJet: Jet Cands and Counts Output to Global Trigger"),
  pair<unsigned, string>(0x580,"ConcJet: Sort Input"),
  pair<unsigned, string>(0x587,"ConcJet: Bunch Counter Pattern Test"),
  pair<unsigned, string>(0x683,"ConcElec: EM Cands and Energy Sums Output to Global Trigger"),
  pair<unsigned, string>(0x680,"ConcElec: Sort Input"),
  pair<unsigned, string>(0x686,"ConcElec: GT Serdes Loopback"),
  pair<unsigned, string>(0x687,"ConcElec: Bunch Counter Pattern Test"),
  pair<unsigned, string>(0x800,"Leaf0ElecPosEtaU1: Sort Input"),
  pair<unsigned, string>(0x804,"Leaf0ElecPosEtaU1: Raw Input"),
  pair<unsigned, string>(0x803,"Leaf0ElecPosEtaU1: Sort Output"),
  pair<unsigned, string>(0x880,"Leaf0ElecPosEtaU2: Sort Input"),
  pair<unsigned, string>(0x884,"Leaf0ElecPosEtaU2: Raw Input"),
  pair<unsigned, string>(0x883,"Leaf0ElecPosEtaU2: Sort Output"),
  pair<unsigned, string>(0xc00,"Leaf0ElecNegEtaU1: Sort Input"),
  pair<unsigned, string>(0xc04,"Leaf0ElecNegEtaU1: Raw Input"),
  pair<unsigned, string>(0xc03,"Leaf0ElecNegEtaU1: Sort Output"),
  pair<unsigned, string>(0xc80,"Leaf0ElecNegEtaU2: Sort Input"),
  pair<unsigned, string>(0xc84,"Leaf0ElecNegEtaU2: Raw Input"),
  pair<unsigned, string>(0xc83,"Leaf0ElecNegEtaU2: Sort Output"),    // end of defined blocks
  pair<unsigned, string>(0x900,"Leaf1JetPosEtaU1: JF2 Input"),       // START OF POS ETA JET LEAVES
  pair<unsigned, string>(0x904,"Leaf1JetPosEtaU1: JF2 Raw Input"),
  pair<unsigned, string>(0x901,"Leaf1JetPosEtaU1: JF2 Shared Recieved"),
  pair<unsigned, string>(0x902,"Leaf1JetPosEtaU1: JF2 Shared Sent"),
  pair<unsigned, string>(0x903,"Leaf1JetPosEtaU1: JF2 Output"),
  pair<unsigned, string>(0x908,"Leaf1JetPosEtaU1: JF3 Input"),
  pair<unsigned, string>(0x90c,"Leaf1JetPosEtaU1: JF3 Raw Input"),
  pair<unsigned, string>(0x909,"Leaf1JetPosEtaU1: JF3 Shared Recieved"),
  pair<unsigned, string>(0x90a,"Leaf1JetPosEtaU1: JF3 Shared Sent"),
  pair<unsigned, string>(0x90b,"Leaf1JetPosEtaU1: JF3 Output"),
  pair<unsigned, string>(0x980,"Leaf1JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  pair<unsigned, string>(0x984,"Leaf1JetPosEtaU2: Eta0 Raw Input"),
  pair<unsigned, string>(0x988,"Leaf1JetPosEtaU2: JF1 Input"),
  pair<unsigned, string>(0x98c,"Leaf1JetPosEtaU2: JF1 Raw Input"),
  pair<unsigned, string>(0x989,"Leaf1JetPosEtaU2: JF1 Shared Recieved"),
  pair<unsigned, string>(0x98a,"Leaf1JetPosEtaU2: JF1 Shared Sent"),
  pair<unsigned, string>(0x98b,"Leaf1JetPosEtaU2: JF1 Output"),
  pair<unsigned, string>(0xa00,"Leaf2JetPosEtaU1: JF2 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xa04,"Leaf2JetPosEtaU1: JF2 Raw Input"),
  pair<unsigned, string>(0xa01,"Leaf2JetPosEtaU1: JF2 Shared Recieved"),
  pair<unsigned, string>(0xa02,"Leaf2JetPosEtaU1: JF2 Shared Sent"),
  pair<unsigned, string>(0xa03,"Leaf2JetPosEtaU1: JF2 Output"),
  pair<unsigned, string>(0xa08,"Leaf2JetPosEtaU1: JF3 Input"),
  pair<unsigned, string>(0xa0c,"Leaf2JetPosEtaU1: JF3 Raw Input"),
  pair<unsigned, string>(0xa09,"Leaf2JetPosEtaU1: JF3 Shared Recieved"),
  pair<unsigned, string>(0xa0a,"Leaf2JetPosEtaU1: JF3 Shared Sent"),
  pair<unsigned, string>(0xa0b,"Leaf2JetPosEtaU1: JF3 Output"),
  pair<unsigned, string>(0xa80,"Leaf2JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xa84,"Leaf2JetPosEtaU2: Eta0 Raw Input"),
  pair<unsigned, string>(0xa88,"Leaf2JetPosEtaU2: JF1 Input"),
  pair<unsigned, string>(0xa8c,"Leaf2JetPosEtaU2: JF1 Raw Input"),
  pair<unsigned, string>(0xa89,"Leaf2JetPosEtaU2: JF1 Shared Recieved"),
  pair<unsigned, string>(0xa8a,"Leaf2JetPosEtaU2: JF1 Shared Sent"),
  pair<unsigned, string>(0xa8b,"Leaf2JetPosEtaU2: JF1 Output"),
  pair<unsigned, string>(0xb00,"Leaf3JetPosEtaU1: JF2 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xb04,"Leaf3JetPosEtaU1: JF2 Raw Input"),
  pair<unsigned, string>(0xb01,"Leaf3JetPosEtaU1: JF2 Shared Recieved"),
  pair<unsigned, string>(0xb02,"Leaf3JetPosEtaU1: JF2 Shared Sent"),
  pair<unsigned, string>(0xb03,"Leaf3JetPosEtaU1: JF2 Output"),
  pair<unsigned, string>(0xb08,"Leaf3JetPosEtaU1: JF3 Input"),
  pair<unsigned, string>(0xb0c,"Leaf3JetPosEtaU1: JF3 Raw Input"),
  pair<unsigned, string>(0xb09,"Leaf3JetPosEtaU1: JF3 Shared Recieved"),
  pair<unsigned, string>(0xb0a,"Leaf3JetPosEtaU1: JF3 Shared Sent"),
  pair<unsigned, string>(0xb0b,"Leaf3JetPosEtaU1: JF3 Output"),
  pair<unsigned, string>(0xb80,"Leaf3JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xb84,"Leaf3JetPosEtaU2: Eta0 Raw Input"),
  pair<unsigned, string>(0xb88,"Leaf3JetPosEtaU2: JF1 Input"),
  pair<unsigned, string>(0xb8c,"Leaf3JetPosEtaU2: JF1 Raw Input"),
  pair<unsigned, string>(0xb89,"Leaf3JetPosEtaU2: JF1 Shared Recieved"),
  pair<unsigned, string>(0xb8a,"Leaf3JetPosEtaU2: JF1 Shared Sent"),
  pair<unsigned, string>(0xb8b,"Leaf3JetPosEtaU2: JF1 Output"),
  pair<unsigned, string>(0xd00,"Leaf1JetNegEtaU1: JF2 Input"),       // START OF NEG ETA JET LEAVES
  pair<unsigned, string>(0xd04,"Leaf1JetNegEtaU1: JF2 Raw Input"),
  pair<unsigned, string>(0xd01,"Leaf1JetNegEtaU1: JF2 Shared Recieved"),
  pair<unsigned, string>(0xd02,"Leaf1JetNegEtaU1: JF2 Shared Sent"),
  pair<unsigned, string>(0xd03,"Leaf1JetNegEtaU1: JF2 Output"),
  pair<unsigned, string>(0xd08,"Leaf1JetNegEtaU1: JF3 Input"),
  pair<unsigned, string>(0xd0c,"Leaf1JetNegEtaU1: JF3 Raw Input"),
  pair<unsigned, string>(0xd09,"Leaf1JetNegEtaU1: JF3 Shared Recieved"),
  pair<unsigned, string>(0xd0a,"Leaf1JetNegEtaU1: JF3 Shared Sent"),
  pair<unsigned, string>(0xd0b,"Leaf1JetNegEtaU1: JF3 Output"),
  pair<unsigned, string>(0xd80,"Leaf1JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xd84,"Leaf1JetNegEtaU2: Eta0 Raw Input"),
  pair<unsigned, string>(0xd88,"Leaf1JetNegEtaU2: JF1 Input"),
  pair<unsigned, string>(0xd8c,"Leaf1JetNegEtaU2: JF1 Raw Input"),
  pair<unsigned, string>(0xd89,"Leaf1JetNegEtaU2: JF1 Shared Recieved"),
  pair<unsigned, string>(0xd8a,"Leaf1JetNegEtaU2: JF1 Shared Sent"),
  pair<unsigned, string>(0xd8b,"Leaf1JetNegEtaU2: JF1 Output"),
  pair<unsigned, string>(0xe00,"Leaf2JetNegEtaU1: JF2 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xe04,"Leaf2JetNegEtaU1: JF2 Raw Input"),
  pair<unsigned, string>(0xe01,"Leaf2JetNegEtaU1: JF2 Shared Recieved"),
  pair<unsigned, string>(0xe02,"Leaf2JetNegEtaU1: JF2 Shared Sent"),
  pair<unsigned, string>(0xe03,"Leaf2JetNegEtaU1: JF2 Output"),
  pair<unsigned, string>(0xe08,"Leaf2JetNegEtaU1: JF3 Input"),
  pair<unsigned, string>(0xe0c,"Leaf2JetNegEtaU1: JF3 Raw Input"),
  pair<unsigned, string>(0xe09,"Leaf2JetNegEtaU1: JF3 Shared Recieved"),
  pair<unsigned, string>(0xe0a,"Leaf2JetNegEtaU1: JF3 Shared Sent"),
  pair<unsigned, string>(0xe0b,"Leaf2JetNegEtaU1: JF3 Output"),
  pair<unsigned, string>(0xe80,"Leaf2JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xe84,"Leaf2JetNegEtaU2: Eta0 Raw Input"),
  pair<unsigned, string>(0xe88,"Leaf2JetNegEtaU2: JF1 Input"),
  pair<unsigned, string>(0xe8c,"Leaf2JetNegEtaU2: JF1 Raw Input"),
  pair<unsigned, string>(0xe89,"Leaf2JetNegEtaU2: JF1 Shared Recieved"),
  pair<unsigned, string>(0xe8a,"Leaf2JetNegEtaU2: JF1 Shared Sent"),
  pair<unsigned, string>(0xe8b,"Leaf2JetNegEtaU2: JF1 Output"),
  pair<unsigned, string>(0xf00,"Leaf3JetNegEtaU1: JF2 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xf04,"Leaf3JetNegEtaU1: JF2 Raw Input"),
  pair<unsigned, string>(0xf01,"Leaf3JetNegEtaU1: JF2 Shared Recieved"),
  pair<unsigned, string>(0xf02,"Leaf3JetNegEtaU1: JF2 Shared Sent"),
  pair<unsigned, string>(0xf03,"Leaf3JetNegEtaU1: JF2 Output"),
  pair<unsigned, string>(0xf08,"Leaf3JetNegEtaU1: JF3 Input"),
  pair<unsigned, string>(0xf0c,"Leaf3JetNegEtaU1: JF3 Raw Input"),
  pair<unsigned, string>(0xf09,"Leaf3JetNegEtaU1: JF3 Shared Recieved"),
  pair<unsigned, string>(0xf0a,"Leaf3JetNegEtaU1: JF3 Shared Sent"),
  pair<unsigned, string>(0xf0b,"Leaf3JetNegEtaU1: JF3 Output"),
  pair<unsigned, string>(0xf80,"Leaf3JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  pair<unsigned, string>(0xf84,"Leaf3JetNegEtaU2: Eta0 Raw Input"),
  pair<unsigned, string>(0xf88,"Leaf3JetNegEtaU2: JF1 Input"),
  pair<unsigned, string>(0xf8c,"Leaf3JetNegEtaU2: JF1 Raw Input"),
  pair<unsigned, string>(0xf89,"Leaf3JetNegEtaU2: JF1 Shared Recieved"),
  pair<unsigned, string>(0xf8a,"Leaf3JetNegEtaU2: JF1 Shared Sent"),
  pair<unsigned, string>(0xf8b,"Leaf3JetNegEtaU2: JF1 Output")
//  pair<unsigned, string>(0x300,"WheelPosEtaJet: Input"),
//  pair<unsigned, string>(0x303,"WheelPosEtaJet: Output"),
//  pair<unsigned, string>(0x380,"WheelPosEtaEnergy: Input"),
//  pair<unsigned, string>(0x383,"WheelPosEtaEnergy: Output"),
//  pair<unsigned, string>(0x700,"WheelNegEtaJet: Input"),
//  pair<unsigned, string>(0x703,"WheelNegEtaJet: Output"),
//  pair<unsigned, string>(0x780,"WheelNegEtaEnergy: Input"),
//  pair<unsigned, string>(0x783,"WheelNegEtaEnergy: Output")
};

map<unsigned, string> GctBlockHeader::blockName_(b, b + sizeof(b) / sizeof(b[0]));

