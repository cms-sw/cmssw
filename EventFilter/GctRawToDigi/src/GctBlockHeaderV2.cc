#include "EventFilter/GctRawToDigi/src/GctBlockHeaderV2.h"

using std::string;

// Initialise the static block header length map
GctBlockHeaderV2::BlockLengthMap GctBlockHeaderV2::blockLengthV2_();


// CONSTRUCTOR
GctBlockHeaderV2::GctBlockHeaderV2(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid):
  GctBlockHeaderBase()
{
  d = (id & 0xfff) + ((nsamples&0xf)<<16) + ((bcid&0xfff)<<20) + ((evid&0xf)<<12);
}


void GctBlockHeaderV2::initBlockLengthMap(const BlockLengthMapVersion lengthMapVersion)
{
  blockLengthMap().clear();
  
  
  // ** FIRST MAKE BLOCK-ID => BLOCK-LENGTH MAPPINGS COMMON TO ALL LENGTH MAP VERSIONS **
  
  // MISCELLANEOUS BLOCKS
  blockLengthMap().insert(make_pair(0x000,0));      // NULL
  blockLengthMap().insert(make_pair(0x0ff,198));    // Temporary hack: All RCT Calo Regions for CMSSW pack/unpack
  // CONC JET FPGA
  blockLengthMap().insert(make_pair(0x580,12));     // ConcJet: Input TrigPathA (Jet Cands)
  blockLengthMap().insert(make_pair(0x581,2));      // ConcJet: Input TrigPathB (HF Rings)
  blockLengthMap().insert(make_pair(0x582,4));      // ConcJet: Input TrigPathC (MissHt)
  blockLengthMap().insert(make_pair(0x583,8));      // ConcJet: Jet Cands and Counts Output to GT
  blockLengthMap().insert(make_pair(0x587,4));      // ConcJet: BX & Orbit Info
  // CONC ELEC FPGA
  blockLengthMap().insert(make_pair(0x680,16));     // ConcElec: Input TrigPathA (EM Cands)
  blockLengthMap().insert(make_pair(0x681,6));      // ConcElec: Input TrigPathB (Et Sums)
  blockLengthMap().insert(make_pair(0x682,2));      // ConcElec: Input TrigPathC (Ht Sums)
  blockLengthMap().insert(make_pair(0x683,6));      // ConcElec: EM Cands and Energy Sums Output to GT
  blockLengthMap().insert(make_pair(0x686,2));      // ConcElec: Test (GT Serdes Loopback)
  blockLengthMap().insert(make_pair(0x687,4));      // ConcElec: BX & Orbit Info
  // ELECTRON LEAF FPGAS
  blockLengthMap().insert(make_pair(0x800,20));     // Leaf0ElecPosEtaU1: Sort Input
  blockLengthMap().insert(make_pair(0x803,4));      // Leaf0ElecPosEtaU1: Sort Output
  blockLengthMap().insert(make_pair(0x804,15));     // Leaf0ElecPosEtaU1: Raw Input
  blockLengthMap().insert(make_pair(0x880,16));     // Leaf0ElecPosEtaU2: Sort Input
  blockLengthMap().insert(make_pair(0x883,4));      // Leaf0ElecPosEtaU2: Sort Output
  blockLengthMap().insert(make_pair(0x884,12));     // Leaf0ElecPosEtaU2: Raw Input
  blockLengthMap().insert(make_pair(0xc00,20));     // Leaf0ElecNegEtaU1: Sort Input
  blockLengthMap().insert(make_pair(0xc03,4));      // Leaf0ElecNegEtaU1: Sort Output
  blockLengthMap().insert(make_pair(0xc04,15));     // Leaf0ElecNegEtaU1: Raw Input
  blockLengthMap().insert(make_pair(0xc80,16));     // Leaf0ElecNegEtaU2: Sort Input
  blockLengthMap().insert(make_pair(0xc83,4));      // Leaf0ElecNegEtaU2: Sort Output
  blockLengthMap().insert(make_pair(0xc84,12));     // Leaf0ElecNegEtaU2: Raw Input
  // WHEEL POS ETA JET FPGA
  blockLengthMap().insert(make_pair(0x300,27));     // WheelPosEtaJet: Input TrigPathA (Jet Sort)
  blockLengthMap().insert(make_pair(0x301,3));      // WheelPosEtaJet: Input TrigPathB (MissHt)
  blockLengthMap().insert(make_pair(0x303,6));      // WheelPosEtaJet: Output TrigPathA (Jet Sort)
  blockLengthMap().insert(make_pair(0x305,2));      // WheelPosEtaJet: Output TrigPathB (MissHt)
  blockLengthMap().insert(make_pair(0x306,32));     // WheelPosEtaJet: Test (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
  blockLengthMap().insert(make_pair(0x307,4));      // WheelPosEtaJet: Info (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
  // WHEEL POS ETA ENERGY FPGA
  blockLengthMap().insert(make_pair(0x380,21));     // WheelPosEtaEnergy: Input TrigPathA (Et)
  blockLengthMap().insert(make_pair(0x381,3));      // WheelPosEtaEnergy: Input TrigPathB (Ht)
  blockLengthMap().insert(make_pair(0x383,7));      // WheelPosEtaEnergy: Output TrigPathA (Et)
  blockLengthMap().insert(make_pair(0x385,2));      // WheelPosEtaEnergy: Output TrigPathB (Ht)
  blockLengthMap().insert(make_pair(0x386,32));     // WheelPosEtaEnergy: Test
  blockLengthMap().insert(make_pair(0x387,6));      // WheelPosEtaEnergy: BX & Orbit Info   (Potential data incompatibility between V24/V25 where block length=4, and V27.1 where block length=6)
  // WHEEL NEG ETA JET FPGA
  blockLengthMap().insert(make_pair(0x700,27));     // WheelNegEtaJet: Input TrigPathA (Jet Sort)
  blockLengthMap().insert(make_pair(0x701,3));      // WheelNegEtaJet: Input TrigPathB (MissHt)
  blockLengthMap().insert(make_pair(0x703,6));      // WheelNegEtaJet: Output TrigPathA (Jet Sort)
  blockLengthMap().insert(make_pair(0x705,2));      // WheelNegEtaJet: Output TrigPathB (MissHt)
  blockLengthMap().insert(make_pair(0x706,32));     // WheelNegEtaJet: Test (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
  blockLengthMap().insert(make_pair(0x707,4));      // WheelNegEtaJet: Info (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
  // WHEEL NEG ETA ENERGY FPGA
  blockLengthMap().insert(make_pair(0x780,21));     // WheelNegEtaEnergy: Input TrigPathA (Et)
  blockLengthMap().insert(make_pair(0x781,3));      // WheelNegEtaEnergy: Input TrigPathB (Ht)
  blockLengthMap().insert(make_pair(0x783,7));      // WheelNegEtaEnergy: Output TrigPathA (Et)
  blockLengthMap().insert(make_pair(0x785,2));      // WheelNegEtaEnergy: Output TrigPathB (Ht)
  blockLengthMap().insert(make_pair(0x786,32));     // WheelNegEtaEnergy: Test
  blockLengthMap().insert(make_pair(0x787,6));      // WheelNegEtaEnergy: BX & Orbit Info   (Potential data incompatibility between V24/V25 where block length=4, and V27.1 where block length=6)
  // JET LEAF FPGAS - POSITIVE ETA
  blockLengthMap().insert(make_pair(0x901,3));      // Leaf1JetPosEtaU1: JF2 Shared Received
  blockLengthMap().insert(make_pair(0x902,3));      // Leaf1JetPosEtaU1: JF2 Shared Sent
  blockLengthMap().insert(make_pair(0x903,10));     // Leaf1JetPosEtaU1: JF2 Output
  blockLengthMap().insert(make_pair(0x904,8));      // Leaf1JetPosEtaU1: JF2 Raw Input
  blockLengthMap().insert(make_pair(0x909,3));      // Leaf1JetPosEtaU1: JF3 Shared Received
  blockLengthMap().insert(make_pair(0x90a,3));      // Leaf1JetPosEtaU1: JF3 Shared Sent
  blockLengthMap().insert(make_pair(0x90b,10));     // Leaf1JetPosEtaU1: JF3 Output
  blockLengthMap().insert(make_pair(0x90c,8));      // Leaf1JetPosEtaU1: JF3 Raw Input
  blockLengthMap().insert(make_pair(0x980,3));      // Leaf1JetPosEtaU2: Eta0 Input
  blockLengthMap().insert(make_pair(0x984,6));      // Leaf1JetPosEtaU2: Eta0 Raw Input
  blockLengthMap().insert(make_pair(0x989,3));      // Leaf1JetPosEtaU2: JF1 Shared Received
  blockLengthMap().insert(make_pair(0x98a,3));      // Leaf1JetPosEtaU2: JF1 Shared Sent
  blockLengthMap().insert(make_pair(0x98b,10));     // Leaf1JetPosEtaU2: JF1 Output
  blockLengthMap().insert(make_pair(0x98c,8));      // Leaf1JetPosEtaU2: JF1 Raw Input
  blockLengthMap().insert(make_pair(0xa01,3));      // Leaf2JetPosEtaU1: JF2 Shared Received
  blockLengthMap().insert(make_pair(0xa02,3));      // Leaf2JetPosEtaU1: JF2 Shared Sent
  blockLengthMap().insert(make_pair(0xa03,10));     // Leaf2JetPosEtaU1: JF2 Output
  blockLengthMap().insert(make_pair(0xa04,8));      // Leaf2JetPosEtaU1: JF2 Raw Input
  blockLengthMap().insert(make_pair(0xa09,3));      // Leaf2JetPosEtaU1: JF3 Shared Received
  blockLengthMap().insert(make_pair(0xa0a,3));      // Leaf2JetPosEtaU1: JF3 Shared Sent
  blockLengthMap().insert(make_pair(0xa0b,10));     // Leaf2JetPosEtaU1: JF3 Output
  blockLengthMap().insert(make_pair(0xa0c,8));      // Leaf2JetPosEtaU1: JF3 Raw Input
  blockLengthMap().insert(make_pair(0xa80,3));      // Leaf2JetPosEtaU2: Eta0 Input
  blockLengthMap().insert(make_pair(0xa84,6));      // Leaf2JetPosEtaU2: Eta0 Raw Input
  blockLengthMap().insert(make_pair(0xa89,3));      // Leaf2JetPosEtaU2: JF1 Shared Received
  blockLengthMap().insert(make_pair(0xa8a,3));      // Leaf2JetPosEtaU2: JF1 Shared Sent
  blockLengthMap().insert(make_pair(0xa8b,10));     // Leaf2JetPosEtaU2: JF1 Output
  blockLengthMap().insert(make_pair(0xa8c,8));      // Leaf2JetPosEtaU2: JF1 Raw Input
  blockLengthMap().insert(make_pair(0xb01,3));      // Leaf3JetPosEtaU1: JF2 Shared Received
  blockLengthMap().insert(make_pair(0xb02,3));      // Leaf3JetPosEtaU1: JF2 Shared Sent
  blockLengthMap().insert(make_pair(0xb03,10));     // Leaf3JetPosEtaU1: JF2 Output
  blockLengthMap().insert(make_pair(0xb04,8));      // Leaf3JetPosEtaU1: JF2 Raw Input
  blockLengthMap().insert(make_pair(0xb09,3));      // Leaf3JetPosEtaU1: JF3 Shared Received
  blockLengthMap().insert(make_pair(0xb0a,3));      // Leaf3JetPosEtaU1: JF3 Shared Sent
  blockLengthMap().insert(make_pair(0xb0b,10));     // Leaf3JetPosEtaU1: JF3 Output
  blockLengthMap().insert(make_pair(0xb0c,8));      // Leaf3JetPosEtaU1: JF3 Raw Input
  blockLengthMap().insert(make_pair(0xb80,3));      // Leaf3JetPosEtaU2: Eta0 Input
  blockLengthMap().insert(make_pair(0xb84,6));      // Leaf3JetPosEtaU2: Eta0 Raw Input
  blockLengthMap().insert(make_pair(0xb89,3));      // Leaf3JetPosEtaU2: JF1 Shared Received
  blockLengthMap().insert(make_pair(0xb8a,3));      // Leaf3JetPosEtaU2: JF1 Shared Sent
  blockLengthMap().insert(make_pair(0xb8b,10));     // Leaf3JetPosEtaU2: JF1 Output
  blockLengthMap().insert(make_pair(0xb8c,8));      // Leaf3JetPosEtaU2: JF1 Raw Input
  // JET LEAF FPGAS - NEGATIVE ETA
  blockLengthMap().insert(make_pair(0xd01,3));      // Leaf1JetNegEtaU1: JF2 Shared Received
  blockLengthMap().insert(make_pair(0xd02,3));      // Leaf1JetNegEtaU1: JF2 Shared Sent
  blockLengthMap().insert(make_pair(0xd03,10));     // Leaf1JetNegEtaU1: JF2 Output
  blockLengthMap().insert(make_pair(0xd04,8));      // Leaf1JetNegEtaU1: JF2 Raw Input
  blockLengthMap().insert(make_pair(0xd09,3));      // Leaf1JetNegEtaU1: JF3 Shared Received
  blockLengthMap().insert(make_pair(0xd0a,3));      // Leaf1JetNegEtaU1: JF3 Shared Sent
  blockLengthMap().insert(make_pair(0xd0b,10));     // Leaf1JetNegEtaU1: JF3 Output
  blockLengthMap().insert(make_pair(0xd0c,8));      // Leaf1JetNegEtaU1: JF3 Raw Input
  blockLengthMap().insert(make_pair(0xd80,3));      // Leaf1JetNegEtaU2: Eta0 Input
  blockLengthMap().insert(make_pair(0xd84,6));      // Leaf1JetNegEtaU2: Eta0 Raw Input
  blockLengthMap().insert(make_pair(0xd89,3));      // Leaf1JetNegEtaU2: JF1 Shared Received
  blockLengthMap().insert(make_pair(0xd8a,3));      // Leaf1JetNegEtaU2: JF1 Shared Sent
  blockLengthMap().insert(make_pair(0xd8b,10));     // Leaf1JetNegEtaU2: JF1 Output
  blockLengthMap().insert(make_pair(0xd8c,8));      // Leaf1JetNegEtaU2: JF1 Raw Input
  blockLengthMap().insert(make_pair(0xe01,3));      // Leaf2JetNegEtaU1: JF2 Shared Received
  blockLengthMap().insert(make_pair(0xe02,3));      // Leaf2JetNegEtaU1: JF2 Shared Sent
  blockLengthMap().insert(make_pair(0xe03,10));     // Leaf2JetNegEtaU1: JF2 Output
  blockLengthMap().insert(make_pair(0xe04,8));      // Leaf2JetNegEtaU1: JF2 Raw Input
  blockLengthMap().insert(make_pair(0xe09,3));      // Leaf2JetNegEtaU1: JF3 Shared Received
  blockLengthMap().insert(make_pair(0xe0a,3));      // Leaf2JetNegEtaU1: JF3 Shared Sent
  blockLengthMap().insert(make_pair(0xe0b,10));     // Leaf2JetNegEtaU1: JF3 Output
  blockLengthMap().insert(make_pair(0xe0c,8));      // Leaf2JetNegEtaU1: JF3 Raw Input
  blockLengthMap().insert(make_pair(0xe80,3));      // Leaf2JetNegEtaU2: Eta0 Input
  blockLengthMap().insert(make_pair(0xe84,6));      // Leaf2JetNegEtaU2: Eta0 Raw Input
  blockLengthMap().insert(make_pair(0xe89,3));      // Leaf2JetNegEtaU2: JF1 Shared Received
  blockLengthMap().insert(make_pair(0xe8a,3));      // Leaf2JetNegEtaU2: JF1 Shared Sent
  blockLengthMap().insert(make_pair(0xe8b,10));     // Leaf2JetNegEtaU2: JF1 Output
  blockLengthMap().insert(make_pair(0xe8c,8));      // Leaf2JetNegEtaU2: JF1 Raw Input
  blockLengthMap().insert(make_pair(0xf01,3));      // Leaf3JetNegEtaU1: JF2 Shared Received
  blockLengthMap().insert(make_pair(0xf02,3));      // Leaf3JetNegEtaU1: JF2 Shared Sent
  blockLengthMap().insert(make_pair(0xf03,10));     // Leaf3JetNegEtaU1: JF2 Output
  blockLengthMap().insert(make_pair(0xf04,8));      // Leaf3JetNegEtaU1: JF2 Raw Input
  blockLengthMap().insert(make_pair(0xf09,3));      // Leaf3JetNegEtaU1: JF3 Shared Received
  blockLengthMap().insert(make_pair(0xf0a,3));      // Leaf3JetNegEtaU1: JF3 Shared Sent
  blockLengthMap().insert(make_pair(0xf0b,10));     // Leaf3JetNegEtaU1: JF3 Output
  blockLengthMap().insert(make_pair(0xf0c,8));      // Leaf3JetNegEtaU1: JF3 Raw Input
  blockLengthMap().insert(make_pair(0xf80,3));      // Leaf3JetNegEtaU2: Eta0 Input
  blockLengthMap().insert(make_pair(0xf84,6));      // Leaf3JetNegEtaU2: Eta0 Raw Input
  blockLengthMap().insert(make_pair(0xf89,3));      // Leaf3JetNegEtaU2: JF1 Shared Received
  blockLengthMap().insert(make_pair(0xf8a,3));      // Leaf3JetNegEtaU2: JF1 Shared Sent
  blockLengthMap().insert(make_pair(0xf8b,10));     // Leaf3JetNegEtaU2: JF1 Output
  blockLengthMap().insert(make_pair(0xf8c,8));      // Leaf3JetNegEtaU2: JF1 Raw Input


  // ** NOW DO BLOCK-ID => BLOCK-LENGTH MAPPINGS SPECIFIC TO EACH PARTICULAR LENGTH MAP VERSION **

  if(lengthMapVersion == BLOCK_LENGTHS_FOR_UNPACKER_V2)
  {
    // JET LEAF FPGAS - POSITIVE ETA
    blockLengthMap().insert(make_pair(0x900,12));     // Leaf1JetPosEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0x908,12));     // Leaf1JetPosEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0x988,12));     // Leaf1JetPosEtaU2: JF1 Input
    blockLengthMap().insert(make_pair(0xa00,12));     // Leaf2JetPosEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xa08,12));     // Leaf2JetPosEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xa88,12));     // Leaf2JetPosEtaU2: JF1 Input
    blockLengthMap().insert(make_pair(0xb00,12));     // Leaf3JetPosEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xb08,12));     // Leaf3JetPosEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xb88,12));     // Leaf3JetPosEtaU2: JF1 Input
    // JET LEAF FPGAS - NEGATIVE ETA
    blockLengthMap().insert(make_pair(0xd00,12));     // Leaf1JetNegEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xd08,12));     // Leaf1JetNegEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xd88,12));     // Leaf1JetNegEtaU2: JF1 Input
    blockLengthMap().insert(make_pair(0xe00,12));     // Leaf2JetNegEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xe08,12));     // Leaf2JetNegEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xe88,12));     // Leaf2JetNegEtaU2: JF1 Input
    blockLengthMap().insert(make_pair(0xf00,12));     // Leaf3JetNegEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xf08,12));     // Leaf3JetNegEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xf88,12));     // Leaf3JetNegEtaU2: JF1 Input    
  }
  else if(lengthMapVersion == BLOCK_LENGTHS_FOR_UNPACKER_V3)
  {
    // JET LEAF FPGAS - POSITIVE ETA
    blockLengthMap().insert(make_pair(0x900,13));     // Leaf1JetPosEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0x908,13));     // Leaf1JetPosEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0x988,13));     // Leaf1JetPosEtaU2: JF1 Input
    blockLengthMap().insert(make_pair(0xa00,13));     // Leaf2JetPosEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xa08,13));     // Leaf2JetPosEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xa88,13));     // Leaf2JetPosEtaU2: JF1 Input
    blockLengthMap().insert(make_pair(0xb00,13));     // Leaf3JetPosEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xb08,13));     // Leaf3JetPosEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xb88,13));     // Leaf3JetPosEtaU2: JF1 Input
    // JET LEAF FPGAS - NEGATIVE ETA
    blockLengthMap().insert(make_pair(0xd00,13));     // Leaf1JetNegEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xd08,13));     // Leaf1JetNegEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xd88,13));     // Leaf1JetNegEtaU2: JF1 Input
    blockLengthMap().insert(make_pair(0xe00,13));     // Leaf2JetNegEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xe08,13));     // Leaf2JetNegEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xe88,13));     // Leaf2JetNegEtaU2: JF1 Input
    blockLengthMap().insert(make_pair(0xf00,13));     // Leaf3JetNegEtaU1: JF2 Input
    blockLengthMap().insert(make_pair(0xf08,13));     // Leaf3JetNegEtaU1: JF3 Input
    blockLengthMap().insert(make_pair(0xf88,13));     // Leaf3JetNegEtaU2: JF1 Input    
  }
}

/// setup class static to lookup block name
GctBlockHeaderBase::BlockNamePair initArray4[] = {
  // MISCELLANEOUS BLOCKS
  GctBlockHeaderBase::BlockNamePair(0x000,"NULL"),
  GctBlockHeaderBase::BlockNamePair(0x0ff,"All RCT Calo Regions"),  // Temporary hack: All RCT Calo Regions for CMSSW pack/unpack
  // CONC JET FPGA
  GctBlockHeaderBase::BlockNamePair(0x580,"ConcJet: Input TrigPathA (Jet Cands)"),
  GctBlockHeaderBase::BlockNamePair(0x581,"ConcJet: Input TrigPathB (HF Rings)"),
  GctBlockHeaderBase::BlockNamePair(0x582,"ConcJet: Input TrigPathC (MissHt)"),
  GctBlockHeaderBase::BlockNamePair(0x583,"ConcJet: Jet Cands and Counts Output to GT"),
  GctBlockHeaderBase::BlockNamePair(0x587,"ConcJet: BX & Orbit Info"),
  // CONC ELEC FPGA
  GctBlockHeaderBase::BlockNamePair(0x680,"ConcElec: Input TrigPathA (EM Cands)"),
  GctBlockHeaderBase::BlockNamePair(0x681,"ConcElec: Input TrigPathB (Et Sums)"),
  GctBlockHeaderBase::BlockNamePair(0x682,"ConcElec: Input TrigPathC (Ht Sums)"),
  GctBlockHeaderBase::BlockNamePair(0x683,"ConcElec: EM Cands and Energy Sums Output to GT"),
  GctBlockHeaderBase::BlockNamePair(0x686,"ConcElec: Test (GT Serdes Loopback)"),
  GctBlockHeaderBase::BlockNamePair(0x687,"ConcElec: BX & Orbit Info"),
  // ELECTRON LEAF FPGAS
  GctBlockHeaderBase::BlockNamePair(0x800,"Leaf0ElecPosEtaU1: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x803,"Leaf0ElecPosEtaU1: Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0x804,"Leaf0ElecPosEtaU1: Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x880,"Leaf0ElecPosEtaU2: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0x883,"Leaf0ElecPosEtaU2: Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0x884,"Leaf0ElecPosEtaU2: Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xc00,"Leaf0ElecNegEtaU1: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0xc03,"Leaf0ElecNegEtaU1: Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0xc04,"Leaf0ElecNegEtaU1: Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xc80,"Leaf0ElecNegEtaU2: Sort Input"),
  GctBlockHeaderBase::BlockNamePair(0xc83,"Leaf0ElecNegEtaU2: Sort Output"),
  GctBlockHeaderBase::BlockNamePair(0xc84,"Leaf0ElecNegEtaU2: Raw Input"),
  // WHEEL POS ETA JET FPGA
  GctBlockHeaderBase::BlockNamePair(0x300,"WheelPosEtaJet: Input TrigPathA (Jet Sort)"),
  GctBlockHeaderBase::BlockNamePair(0x301,"WheelPosEtaJet: Input TrigPathB (MissHt)"),  
  GctBlockHeaderBase::BlockNamePair(0x303,"WheelPosEtaJet: Output TrigPathA (Jet Sort)"),
  GctBlockHeaderBase::BlockNamePair(0x305,"WheelPosEtaJet: Output TrigPathB (MissHt)"),
  GctBlockHeaderBase::BlockNamePair(0x306,"WheelPosEtaJet: Test (deprecated)"),  // (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
  GctBlockHeaderBase::BlockNamePair(0x307,"WheelPosEtaJet: Info (deprecated)"),  // (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
  // WHEEL POS ETA ENERGY FPGA            "
  GctBlockHeaderBase::BlockNamePair(0x380,"WheelPosEtaEnergy: Input TrigPathA (Et)"),
  GctBlockHeaderBase::BlockNamePair(0x381,"WheelPosEtaEnergy: Input TrigPathB (Ht)"),
  GctBlockHeaderBase::BlockNamePair(0x383,"WheelPosEtaEnergy: Output TrigPathA (Et)"),
  GctBlockHeaderBase::BlockNamePair(0x385,"WheelPosEtaEnergy: Output TrigPathB (Ht)"),
  GctBlockHeaderBase::BlockNamePair(0x386,"WheelPosEtaEnergy: Test"),
  GctBlockHeaderBase::BlockNamePair(0x387,"WheelPosEtaEnergy: BX & Orbit Info"),
  // WHEEL NEG ETA JET FPGA               "
  GctBlockHeaderBase::BlockNamePair(0x700,"WheelNegEtaJet: Input TrigPathA (Jet Sort)"),
  GctBlockHeaderBase::BlockNamePair(0x701,"WheelNegEtaJet: Input TrigPathB (MissHt)"),
  GctBlockHeaderBase::BlockNamePair(0x703,"WheelNegEtaJet: Output TrigPathA (Jet Sort)"),
  GctBlockHeaderBase::BlockNamePair(0x705,"WheelNegEtaJet: Output TrigPathB (MissHt)"),
  GctBlockHeaderBase::BlockNamePair(0x706,"WheelNegEtaJet: Test (deprecated)"),  // (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
  GctBlockHeaderBase::BlockNamePair(0x707,"WheelNegEtaJet: Info (deprecated)"),  // (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
  // WHEEL NEG ETA ENERGY FPGA            "
  GctBlockHeaderBase::BlockNamePair(0x780,"WheelNegEtaEnergy: Input TrigPathA (Et)"),
  GctBlockHeaderBase::BlockNamePair(0x781,"WheelNegEtaEnergy: Input TrigPathB (Ht)"),
  GctBlockHeaderBase::BlockNamePair(0x783,"WheelNegEtaEnergy: Output TrigPathA (Et)"),
  GctBlockHeaderBase::BlockNamePair(0x785,"WheelNegEtaEnergy: Output TrigPathB (Ht)"),
  GctBlockHeaderBase::BlockNamePair(0x786,"WheelNegEtaEnergy: Test"),
  GctBlockHeaderBase::BlockNamePair(0x787,"WheelNegEtaEnergy: BX & Orbit Info"),
  // JET LEAF FPGAS - POSITIVE ETA
  GctBlockHeaderBase::BlockNamePair(0x900,"Leaf1JetPosEtaU1: JF2 Input"),
  GctBlockHeaderBase::BlockNamePair(0x901,"Leaf1JetPosEtaU1: JF2 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0x902,"Leaf1JetPosEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0x903,"Leaf1JetPosEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0x904,"Leaf1JetPosEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x908,"Leaf1JetPosEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0x909,"Leaf1JetPosEtaU1: JF3 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0x90a,"Leaf1JetPosEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0x90b,"Leaf1JetPosEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0x90c,"Leaf1JetPosEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x980,"Leaf1JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0x984,"Leaf1JetPosEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0x988,"Leaf1JetPosEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0x989,"Leaf1JetPosEtaU2: JF1 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0x98a,"Leaf1JetPosEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0x98b,"Leaf1JetPosEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0x98c,"Leaf1JetPosEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xa00,"Leaf2JetPosEtaU1: JF2 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xa01,"Leaf2JetPosEtaU1: JF2 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xa02,"Leaf2JetPosEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xa03,"Leaf2JetPosEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xa04,"Leaf2JetPosEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xa08,"Leaf2JetPosEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xa09,"Leaf2JetPosEtaU1: JF3 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xa0a,"Leaf2JetPosEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xa0b,"Leaf2JetPosEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xa0c,"Leaf2JetPosEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xa80,"Leaf2JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xa84,"Leaf2JetPosEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xa88,"Leaf2JetPosEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xa89,"Leaf2JetPosEtaU2: JF1 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xa8a,"Leaf2JetPosEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xa8b,"Leaf2JetPosEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xa8c,"Leaf2JetPosEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xb00,"Leaf3JetPosEtaU1: JF2 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xb01,"Leaf3JetPosEtaU1: JF2 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xb02,"Leaf3JetPosEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xb03,"Leaf3JetPosEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xb04,"Leaf3JetPosEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xb08,"Leaf3JetPosEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xb09,"Leaf3JetPosEtaU1: JF3 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xb0a,"Leaf3JetPosEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xb0b,"Leaf3JetPosEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xb0c,"Leaf3JetPosEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xb80,"Leaf3JetPosEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xb84,"Leaf3JetPosEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xb88,"Leaf3JetPosEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xb89,"Leaf3JetPosEtaU2: JF1 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xb8a,"Leaf3JetPosEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xb8b,"Leaf3JetPosEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xb8c,"Leaf3JetPosEtaU2: JF1 Raw Input"),
  // JET LEAF FPGAS - NEGATIVE ETA
  GctBlockHeaderBase::BlockNamePair(0xd00,"Leaf1JetNegEtaU1: JF2 Input"),       // START OF NEG ETA JET LEAVES
  GctBlockHeaderBase::BlockNamePair(0xd01,"Leaf1JetNegEtaU1: JF2 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xd02,"Leaf1JetNegEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xd03,"Leaf1JetNegEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xd04,"Leaf1JetNegEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xd08,"Leaf1JetNegEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xd09,"Leaf1JetNegEtaU1: JF3 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xd0a,"Leaf1JetNegEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xd0b,"Leaf1JetNegEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xd0c,"Leaf1JetNegEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xd80,"Leaf1JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xd84,"Leaf1JetNegEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xd88,"Leaf1JetNegEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xd89,"Leaf1JetNegEtaU2: JF1 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xd8a,"Leaf1JetNegEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xd8b,"Leaf1JetNegEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xd8c,"Leaf1JetNegEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xe00,"Leaf2JetNegEtaU1: JF2 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xe01,"Leaf2JetNegEtaU1: JF2 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xe02,"Leaf2JetNegEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xe03,"Leaf2JetNegEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xe04,"Leaf2JetNegEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xe08,"Leaf2JetNegEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xe09,"Leaf2JetNegEtaU1: JF3 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xe0a,"Leaf2JetNegEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xe0b,"Leaf2JetNegEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xe0c,"Leaf2JetNegEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xe80,"Leaf2JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xe84,"Leaf2JetNegEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xe88,"Leaf2JetNegEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xe89,"Leaf2JetNegEtaU2: JF1 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xe8a,"Leaf2JetNegEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xe8b,"Leaf2JetNegEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xe8c,"Leaf2JetNegEtaU2: JF1 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xf00,"Leaf3JetNegEtaU1: JF2 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xf01,"Leaf3JetNegEtaU1: JF2 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xf02,"Leaf3JetNegEtaU1: JF2 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xf03,"Leaf3JetNegEtaU1: JF2 Output"),
  GctBlockHeaderBase::BlockNamePair(0xf04,"Leaf3JetNegEtaU1: JF2 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xf08,"Leaf3JetNegEtaU1: JF3 Input"),
  GctBlockHeaderBase::BlockNamePair(0xf09,"Leaf3JetNegEtaU1: JF3 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xf0a,"Leaf3JetNegEtaU1: JF3 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xf0b,"Leaf3JetNegEtaU1: JF3 Output"),
  GctBlockHeaderBase::BlockNamePair(0xf0c,"Leaf3JetNegEtaU1: JF3 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xf80,"Leaf3JetNegEtaU2: Eta0 Input"),  // Next Leaf Start
  GctBlockHeaderBase::BlockNamePair(0xf84,"Leaf3JetNegEtaU2: Eta0 Raw Input"),
  GctBlockHeaderBase::BlockNamePair(0xf88,"Leaf3JetNegEtaU2: JF1 Input"),
  GctBlockHeaderBase::BlockNamePair(0xf89,"Leaf3JetNegEtaU2: JF1 Shared Received"),
  GctBlockHeaderBase::BlockNamePair(0xf8a,"Leaf3JetNegEtaU2: JF1 Shared Sent"),
  GctBlockHeaderBase::BlockNamePair(0xf8b,"Leaf3JetNegEtaU2: JF1 Output"),
  GctBlockHeaderBase::BlockNamePair(0xf8c,"Leaf3JetNegEtaU2: JF1 Raw Input")
};

GctBlockHeaderV2::BlockNameMap GctBlockHeaderV2::blockNameV2_(initArray4, initArray4 + sizeof(initArray4) / sizeof(initArray4[0]));

