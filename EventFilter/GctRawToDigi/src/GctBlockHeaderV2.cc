#include "EventFilter/GctRawToDigi/src/GctBlockHeaderV2.h"

using std::string;

GctBlockHeaderV2::GctBlockHeaderV2(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid):
  GctBlockHeaderBase()
{
  d = (id & 0xfff) + ((nsamples&0xf)<<16) + ((bcid&0xfff)<<20) + ((evid&0xf)<<12);
}

/// setup class static to lookup block length
GctBlockHeaderBase::BlockLengthPair initArray3[] = {
  GctBlockHeaderBase::BlockLengthPair(0x000,0),
  GctBlockHeaderBase::BlockLengthPair(0x583,8),
  GctBlockHeaderBase::BlockLengthPair(0x580,12),
  GctBlockHeaderBase::BlockLengthPair(0x587,4),
  GctBlockHeaderBase::BlockLengthPair(0x683,6),
  GctBlockHeaderBase::BlockLengthPair(0x680,16),
  GctBlockHeaderBase::BlockLengthPair(0x686,2),
  GctBlockHeaderBase::BlockLengthPair(0x687,4),
  GctBlockHeaderBase::BlockLengthPair(0x800,20),
  GctBlockHeaderBase::BlockLengthPair(0x804,15),
  GctBlockHeaderBase::BlockLengthPair(0x803,4),
  GctBlockHeaderBase::BlockLengthPair(0x880,16),
  GctBlockHeaderBase::BlockLengthPair(0x884,12),
  GctBlockHeaderBase::BlockLengthPair(0x883,4),
  GctBlockHeaderBase::BlockLengthPair(0xc00,20),
  GctBlockHeaderBase::BlockLengthPair(0xc04,15),
  GctBlockHeaderBase::BlockLengthPair(0xc03,4),
  GctBlockHeaderBase::BlockLengthPair(0xc80,16),
  GctBlockHeaderBase::BlockLengthPair(0xc84,12),
  GctBlockHeaderBase::BlockLengthPair(0xc83,4),      // -- end of strictly defined blocks --
  GctBlockHeaderBase::BlockLengthPair(0x900,12),     // Leaf1JetPosEtaU1: JF2 Input
  GctBlockHeaderBase::BlockLengthPair(0x904,8),      // Leaf1JetPosEtaU1: JF2 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0x901,3),      // Leaf1JetPosEtaU1: JF2 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0x902,3),      // Leaf1JetPosEtaU1: JF2 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0x903,10),     // Leaf1JetPosEtaU1: JF2 Output
  GctBlockHeaderBase::BlockLengthPair(0x908,12),     // Leaf1JetPosEtaU1: JF3 Input
  GctBlockHeaderBase::BlockLengthPair(0x90c,8),      // Leaf1JetPosEtaU1: JF3 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0x909,3),      // Leaf1JetPosEtaU1: JF3 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0x90a,3),      // Leaf1JetPosEtaU1: JF3 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0x90b,10),     // Leaf1JetPosEtaU1: JF3 Output
  GctBlockHeaderBase::BlockLengthPair(0x980,3),      // Leaf1JetPosEtaU2: Eta0 Input
  GctBlockHeaderBase::BlockLengthPair(0x984,6),      // Leaf1JetPosEtaU2: Eta0 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0x988,12),     // Leaf1JetPosEtaU2: JF1 Input
  GctBlockHeaderBase::BlockLengthPair(0x98c,8),      // Leaf1JetPosEtaU2: JF1 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0x989,3),      // Leaf1JetPosEtaU2: JF1 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0x98a,3),      // Leaf1JetPosEtaU2: JF1 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0x98b,10),     // Leaf1JetPosEtaU2: JF1 Output
  GctBlockHeaderBase::BlockLengthPair(0xa00,12),     // Leaf2JetPosEtaU1: JF2 Input
  GctBlockHeaderBase::BlockLengthPair(0xa04,8),      // Leaf2JetPosEtaU1: JF2 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xa01,3),      // Leaf2JetPosEtaU1: JF2 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xa02,3),      // Leaf2JetPosEtaU1: JF2 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xa03,10),     // Leaf2JetPosEtaU1: JF2 Output
  GctBlockHeaderBase::BlockLengthPair(0xa08,12),     // Leaf2JetPosEtaU1: JF3 Input
  GctBlockHeaderBase::BlockLengthPair(0xa0c,8),      // Leaf2JetPosEtaU1: JF3 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xa09,3),      // Leaf2JetPosEtaU1: JF3 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xa0a,3),      // Leaf2JetPosEtaU1: JF3 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xa0b,10),     // Leaf2JetPosEtaU1: JF3 Output
  GctBlockHeaderBase::BlockLengthPair(0xa80,3),      // Leaf2JetPosEtaU2: Eta0 Input
  GctBlockHeaderBase::BlockLengthPair(0xa84,6),      // Leaf2JetPosEtaU2: Eta0 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xa88,12),     // Leaf2JetPosEtaU2: JF1 Input
  GctBlockHeaderBase::BlockLengthPair(0xa8c,8),      // Leaf2JetPosEtaU2: JF1 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xa89,3),      // Leaf2JetPosEtaU2: JF1 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xa8a,3),      // Leaf2JetPosEtaU2: JF1 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xa8b,10),     // Leaf2JetPosEtaU2: JF1 Output
  GctBlockHeaderBase::BlockLengthPair(0xb00,12),     // Leaf3JetPosEtaU1: JF2 Input
  GctBlockHeaderBase::BlockLengthPair(0xb04,8),      // Leaf3JetPosEtaU1: JF2 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xb01,3),      // Leaf3JetPosEtaU1: JF2 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xb02,3),      // Leaf3JetPosEtaU1: JF2 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xb03,10),     // Leaf3JetPosEtaU1: JF2 Output
  GctBlockHeaderBase::BlockLengthPair(0xb08,12),     // Leaf3JetPosEtaU1: JF3 Input
  GctBlockHeaderBase::BlockLengthPair(0xb0c,8),      // Leaf3JetPosEtaU1: JF3 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xb09,3),      // Leaf3JetPosEtaU1: JF3 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xb0a,3),      // Leaf3JetPosEtaU1: JF3 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xb0b,10),     // Leaf3JetPosEtaU1: JF3 Output
  GctBlockHeaderBase::BlockLengthPair(0xb80,3),      // Leaf3JetPosEtaU2: Eta0 Input
  GctBlockHeaderBase::BlockLengthPair(0xb84,6),      // Leaf3JetPosEtaU2: Eta0 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xb88,12),     // Leaf3JetPosEtaU2: JF1 Input
  GctBlockHeaderBase::BlockLengthPair(0xb8c,8),      // Leaf3JetPosEtaU2: JF1 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xb89,3),      // Leaf3JetPosEtaU2: JF1 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xb8a,3),      // Leaf3JetPosEtaU2: JF1 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xb8b,10),     // Leaf3JetPosEtaU2: JF1 Output
  GctBlockHeaderBase::BlockLengthPair(0xd00,12),     // Leaf1JetNegEtaU1: JF2 Input
  GctBlockHeaderBase::BlockLengthPair(0xd04,8),      // Leaf1JetNegEtaU1: JF2 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xd01,3),      // Leaf1JetNegEtaU1: JF2 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xd02,3),      // Leaf1JetNegEtaU1: JF2 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xd03,10),     // Leaf1JetNegEtaU1: JF2 Output
  GctBlockHeaderBase::BlockLengthPair(0xd08,12),     // Leaf1JetNegEtaU1: JF3 Input
  GctBlockHeaderBase::BlockLengthPair(0xd0c,8),      // Leaf1JetNegEtaU1: JF3 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xd09,3),      // Leaf1JetNegEtaU1: JF3 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xd0a,3),      // Leaf1JetNegEtaU1: JF3 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xd0b,10),     // Leaf1JetNegEtaU1: JF3 Output
  GctBlockHeaderBase::BlockLengthPair(0xd80,3),      // Leaf1JetNegEtaU2: Eta0 Input
  GctBlockHeaderBase::BlockLengthPair(0xd84,6),      // Leaf1JetNegEtaU2: Eta0 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xd88,12),     // Leaf1JetNegEtaU2: JF1 Input
  GctBlockHeaderBase::BlockLengthPair(0xd8c,8),      // Leaf1JetNegEtaU2: JF1 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xd89,3),      // Leaf1JetNegEtaU2: JF1 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xd8a,3),      // Leaf1JetNegEtaU2: JF1 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xd8b,10),     // Leaf1JetNegEtaU2: JF1 Output
  GctBlockHeaderBase::BlockLengthPair(0xe00,12),     // Leaf2JetNegEtaU1: JF2 Input
  GctBlockHeaderBase::BlockLengthPair(0xe04,8),      // Leaf2JetNegEtaU1: JF2 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xe01,3),      // Leaf2JetNegEtaU1: JF2 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xe02,3),      // Leaf2JetNegEtaU1: JF2 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xe03,10),     // Leaf2JetNegEtaU1: JF2 Output
  GctBlockHeaderBase::BlockLengthPair(0xe08,12),     // Leaf2JetNegEtaU1: JF3 Input
  GctBlockHeaderBase::BlockLengthPair(0xe0c,8),      // Leaf2JetNegEtaU1: JF3 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xe09,3),      // Leaf2JetNegEtaU1: JF3 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xe0a,3),      // Leaf2JetNegEtaU1: JF3 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xe0b,10),     // Leaf2JetNegEtaU1: JF3 Output
  GctBlockHeaderBase::BlockLengthPair(0xe80,3),      // Leaf2JetNegEtaU2: Eta0 Input
  GctBlockHeaderBase::BlockLengthPair(0xe84,6),      // Leaf2JetNegEtaU2: Eta0 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xe88,12),     // Leaf2JetNegEtaU2: JF1 Input
  GctBlockHeaderBase::BlockLengthPair(0xe8c,8),      // Leaf2JetNegEtaU2: JF1 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xe89,3),      // Leaf2JetNegEtaU2: JF1 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xe8a,3),      // Leaf2JetNegEtaU2: JF1 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xe8b,10),     // Leaf2JetNegEtaU2: JF1 Output
  GctBlockHeaderBase::BlockLengthPair(0xf00,12),     // Leaf3JetNegEtaU1: JF2 Input
  GctBlockHeaderBase::BlockLengthPair(0xf04,8),      // Leaf3JetNegEtaU1: JF2 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xf01,3),      // Leaf3JetNegEtaU1: JF2 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xf02,3),      // Leaf3JetNegEtaU1: JF2 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xf03,10),     // Leaf3JetNegEtaU1: JF2 Output
  GctBlockHeaderBase::BlockLengthPair(0xf08,12),     // Leaf3JetNegEtaU1: JF3 Input
  GctBlockHeaderBase::BlockLengthPair(0xf0c,8),      // Leaf3JetNegEtaU1: JF3 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xf09,3),      // Leaf3JetNegEtaU1: JF3 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xf0a,3),      // Leaf3JetNegEtaU1: JF3 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xf0b,10),     // Leaf3JetNegEtaU1: JF3 Output
  GctBlockHeaderBase::BlockLengthPair(0xf80,3),      // Leaf3JetNegEtaU2: Eta0 Input
  GctBlockHeaderBase::BlockLengthPair(0xf84,6),      // Leaf3JetNegEtaU2: Eta0 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xf88,12),     // Leaf3JetNegEtaU2: JF1 Input
  GctBlockHeaderBase::BlockLengthPair(0xf8c,8),      // Leaf3JetNegEtaU2: JF1 Raw Input
  GctBlockHeaderBase::BlockLengthPair(0xf89,3),      // Leaf3JetNegEtaU2: JF1 Shared Recieved
  GctBlockHeaderBase::BlockLengthPair(0xf8a,3),      // Leaf3JetNegEtaU2: JF1 Shared Sent
  GctBlockHeaderBase::BlockLengthPair(0xf8b,10),     // Leaf3JetNegEtaU2: JF1 Output
//  GctBlockHeaderBase::BlockLengthPair(0x300,),     // -- START OF WHEEL FPGAS
//  GctBlockHeaderBase::BlockLengthPair(0x303,), // Unknown AT 22/4/08 
//  GctBlockHeaderBase::BlockLengthPair(0x380,),
//  GctBlockHeaderBase::BlockLengthPair(0x383,), // Unknown AT 22/4/08 
//  GctBlockHeaderBase::BlockLengthPair(0x700,),
//  GctBlockHeaderBase::BlockLengthPair(0x703,), // Unknown AT 22/4/08 
//  GctBlockHeaderBase::BlockLengthPair(0x780,),
//  GctBlockHeaderBase::BlockLengthPair(0x783,), // Unknown AT 22/4/08     // -- END OF WHEEL FPGAS
  GctBlockHeaderBase::BlockLengthPair(0x306,32),
  GctBlockHeaderBase::BlockLengthPair(0x307,4),
  GctBlockHeaderBase::BlockLengthPair(0x386,32),
  GctBlockHeaderBase::BlockLengthPair(0x387,4),
  GctBlockHeaderBase::BlockLengthPair(0x586,32),
  GctBlockHeaderBase::BlockLengthPair(0x686,32),
  GctBlockHeaderBase::BlockLengthPair(0x706,32),
  GctBlockHeaderBase::BlockLengthPair(0x707,4),
  GctBlockHeaderBase::BlockLengthPair(0x786,32),
  GctBlockHeaderBase::BlockLengthPair(0x787,4),
  GctBlockHeaderBase::BlockLengthPair(0x0ff,198)     // Our temp hack RCT calo block
};

GctBlockHeaderV2::BlockLengthMap GctBlockHeaderV2::blockLengthV2_(initArray3, initArray3 + sizeof(initArray3) / sizeof(initArray3[0]));

/// setup class static to lookup block name
GctBlockHeaderBase::BlockNamePair initArray4[] = {
  GctBlockHeaderBase::BlockNamePair(0x000,"NULL"),
  GctBlockHeaderBase::BlockNamePair(0x583,"ConcJet: Jet Cands and Counts Output to Global Trigger"),
  GctBlockHeaderBase::BlockNamePair(0x580,"ConcJet: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x587,"ConcJet: Bunch Counter Pattern Test"),
  GctBlockHeaderBase::BlockNamePair(0x683,"ConcElec: EM Cands and Energy Sums Output to Global Trigger"),
  GctBlockHeaderBase::BlockNamePair(0x680,"ConcElec: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x686,"ConcElec: GT Serdes Loopback"),
  GctBlockHeaderBase::BlockNamePair(0x687,"ConcElec: Bunch Counter Pattern Test"),
  GctBlockHeaderBase::BlockNamePair(0x800,"Leaf0ElecPosEtaU1: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x804,"Leaf0ElecPosEtaU1: Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x803,"Leaf0ElecPosEtaU1: Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0x880,"Leaf0ElecPosEtaU2: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x884,"Leaf0ElecPosEtaU2: Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x883,"Leaf0ElecPosEtaU2: Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0xc00,"Leaf0ElecNegEtaU1: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0xc04,"Leaf0ElecNegEtaU1: Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xc03,"Leaf0ElecNegEtaU1: Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0xc80,"Leaf0ElecNegEtaU2: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0xc84,"Leaf0ElecNegEtaU2: Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xc83,"Leaf0ElecNegEtaU2: Sort Output"),    // end of defined blocks
  GctBlockHeaderBase::BlockNamePair(0x900,"Leaf1JetPosEtaU1: JF2 Input"),       // START OF POS ETA JET LEAVES
  GctBlockHeaderBase::BlockNamePair(0x904,"Leaf1JetPosEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x901,"Leaf1JetPosEtaU1: JF2 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0x902,"Leaf1JetPosEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0x903,"Leaf1JetPosEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0x908,"Leaf1JetPosEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0x90c,"Leaf1JetPosEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x909,"Leaf1JetPosEtaU1: JF3 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0x90a,"Leaf1JetPosEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0x90b,"Leaf1JetPosEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0x980,"Leaf1JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0x984,"Leaf1JetPosEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x988,"Leaf1JetPosEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0x98c,"Leaf1JetPosEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x989,"Leaf1JetPosEtaU2: JF1 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0x98a,"Leaf1JetPosEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0x98b,"Leaf1JetPosEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xa00,"Leaf2JetPosEtaU1: JF2 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xa04,"Leaf2JetPosEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xa01,"Leaf2JetPosEtaU1: JF2 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xa02,"Leaf2JetPosEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xa03,"Leaf2JetPosEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xa08,"Leaf2JetPosEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xa0c,"Leaf2JetPosEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xa09,"Leaf2JetPosEtaU1: JF3 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xa0a,"Leaf2JetPosEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xa0b,"Leaf2JetPosEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xa80,"Leaf2JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xa84,"Leaf2JetPosEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xa88,"Leaf2JetPosEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xa8c,"Leaf2JetPosEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xa89,"Leaf2JetPosEtaU2: JF1 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xa8a,"Leaf2JetPosEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xa8b,"Leaf2JetPosEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xb00,"Leaf3JetPosEtaU1: JF2 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xb04,"Leaf3JetPosEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xb01,"Leaf3JetPosEtaU1: JF2 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xb02,"Leaf3JetPosEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xb03,"Leaf3JetPosEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xb08,"Leaf3JetPosEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xb0c,"Leaf3JetPosEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xb09,"Leaf3JetPosEtaU1: JF3 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xb0a,"Leaf3JetPosEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xb0b,"Leaf3JetPosEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xb80,"Leaf3JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xb84,"Leaf3JetPosEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xb88,"Leaf3JetPosEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xb8c,"Leaf3JetPosEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xb89,"Leaf3JetPosEtaU2: JF1 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xb8a,"Leaf3JetPosEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xb8b,"Leaf3JetPosEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xd00,"Leaf1JetNegEtaU1: JF2 Input"),       // START OF NEG ETA JET LEAVES
  GctBlockHeaderBase::BlockNamePair(0xd04,"Leaf1JetNegEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xd01,"Leaf1JetNegEtaU1: JF2 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xd02,"Leaf1JetNegEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xd03,"Leaf1JetNegEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xd08,"Leaf1JetNegEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xd0c,"Leaf1JetNegEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xd09,"Leaf1JetNegEtaU1: JF3 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xd0a,"Leaf1JetNegEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xd0b,"Leaf1JetNegEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xd80,"Leaf1JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xd84,"Leaf1JetNegEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xd88,"Leaf1JetNegEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xd8c,"Leaf1JetNegEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xd89,"Leaf1JetNegEtaU2: JF1 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xd8a,"Leaf1JetNegEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xd8b,"Leaf1JetNegEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xe00,"Leaf2JetNegEtaU1: JF2 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xe04,"Leaf2JetNegEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xe01,"Leaf2JetNegEtaU1: JF2 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xe02,"Leaf2JetNegEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xe03,"Leaf2JetNegEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xe08,"Leaf2JetNegEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xe0c,"Leaf2JetNegEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xe09,"Leaf2JetNegEtaU1: JF3 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xe0a,"Leaf2JetNegEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xe0b,"Leaf2JetNegEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xe80,"Leaf2JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xe84,"Leaf2JetNegEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xe88,"Leaf2JetNegEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xe8c,"Leaf2JetNegEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xe89,"Leaf2JetNegEtaU2: JF1 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xe8a,"Leaf2JetNegEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xe8b,"Leaf2JetNegEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xf00,"Leaf3JetNegEtaU1: JF2 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xf04,"Leaf3JetNegEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xf01,"Leaf3JetNegEtaU1: JF2 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xf02,"Leaf3JetNegEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xf03,"Leaf3JetNegEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xf08,"Leaf3JetNegEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xf0c,"Leaf3JetNegEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xf09,"Leaf3JetNegEtaU1: JF3 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xf0a,"Leaf3JetNegEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xf0b,"Leaf3JetNegEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xf80,"Leaf3JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xf84,"Leaf3JetNegEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xf88,"Leaf3JetNegEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xf8c,"Leaf3JetNegEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xf89,"Leaf3JetNegEtaU2: JF1 Shared Recieved"),
  GctBlockHeaderBase::BlockNamePair(0xf8a,"Leaf3JetNegEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xf8b,"Leaf3JetNegEtaU2: JF1 Output"),
//  GctBlockHeaderBase::BlockNamePair(0x300,"WheelPosEtaJet: Input"),
//  GctBlockHeaderBase::BlockNamePair(0x303,"WheelPosEtaJet: Output"),
//  GctBlockHeaderBase::BlockNamePair(0x380,"WheelPosEtaEnergy: Input"),
//  GctBlockHeaderBase::BlockNamePair(0x383,"WheelPosEtaEnergy: Output"),
//  GctBlockHeaderBase::BlockNamePair(0x700,"WheelNegEtaJet: Input"),
//  GctBlockHeaderBase::BlockNamePair(0x703,"WheelNegEtaJet: Output"),
//  GctBlockHeaderBase::BlockNamePair(0x780,"WheelNegEtaEnergy: Input"),
//  GctBlockHeaderBase::BlockNamePair(0x783,"WheelNegEtaEnergy: Output")
  GctBlockHeaderBase::BlockNamePair(0x306,"Test"),
  GctBlockHeaderBase::BlockNamePair(0x307,"Info"),
  GctBlockHeaderBase::BlockNamePair(0x386,"Test"),
  GctBlockHeaderBase::BlockNamePair(0x387,"Info"),
  GctBlockHeaderBase::BlockNamePair(0x586,"Test"),
  GctBlockHeaderBase::BlockNamePair(0x686,"Test"),
  GctBlockHeaderBase::BlockNamePair(0x706,"Test"),
  GctBlockHeaderBase::BlockNamePair(0x707,"Info"),
  GctBlockHeaderBase::BlockNamePair(0x786,"Test"),
  GctBlockHeaderBase::BlockNamePair(0x787,"Info"),
  GctBlockHeaderBase::BlockNamePair(0x0ff,"All RCT Calo Regions")  // Our temp hack RCT calo block
};

GctBlockHeaderV2::BlockNameMap GctBlockHeaderV2::blockNameV2_(initArray4, initArray4 + sizeof(initArray4) / sizeof(initArray4[0]));

