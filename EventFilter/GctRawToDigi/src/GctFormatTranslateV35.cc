#include "EventFilter/GctRawToDigi/src/GctFormatTranslateV35.h"

// C++ headers
#include <iostream>
#include <cassert>

// Framework headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Namespace resolution
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

// INITIALISE STATIC VARIABLES
GctFormatTranslateV35::BlockLengthMap GctFormatTranslateV35::m_blockLength = GctFormatTranslateV35::BlockLengthMap();
GctFormatTranslateV35::BlockNameMap GctFormatTranslateV35::m_blockName = GctFormatTranslateV35::BlockNameMap();
GctFormatTranslateV35::BlockIdToUnpackFnMap GctFormatTranslateV35::m_blockUnpackFn = GctFormatTranslateV35::BlockIdToUnpackFnMap();
GctFormatTranslateV35::BlkToRctCrateMap GctFormatTranslateV35::m_rctEmCrate = GctFormatTranslateV35::BlkToRctCrateMap();
GctFormatTranslateV35::BlkToRctCrateMap GctFormatTranslateV35::m_rctJetCrate = GctFormatTranslateV35::BlkToRctCrateMap();
GctFormatTranslateV35::BlockIdToEmCandIsoBoundMap GctFormatTranslateV35::m_internEmIsoBounds = GctFormatTranslateV35::BlockIdToEmCandIsoBoundMap();


// PUBLIC METHODS

GctFormatTranslateV35::GctFormatTranslateV35(bool hltMode, bool unpackSharedRegions):
  GctFormatTranslateBase(hltMode, unpackSharedRegions)
{
  static bool initClass = true;

  if(initClass)
  {
    initClass = false;

    /*** Setup BlockID to BlockLength Map ***/
    // Miscellaneous Blocks
    m_blockLength.insert(make_pair(0x000,0));      // NULL
    // ConcJet FPGA
    m_blockLength.insert(make_pair(0x580,12));     // ConcJet: Input TrigPathA (Jet Cands)
    m_blockLength.insert(make_pair(0x581,2));      // ConcJet: Input TrigPathB (HF Rings)
    m_blockLength.insert(make_pair(0x583,8));      // ConcJet: Jet Cands and Counts Output to GT
    m_blockLength.insert(make_pair(0x587,4));      // ConcJet: BX & Orbit Info
    // ConcElec FPGA
    m_blockLength.insert(make_pair(0x680,16));     // ConcElec: Input TrigPathA (EM Cands)
    m_blockLength.insert(make_pair(0x681,6));      // ConcElec: Input TrigPathB (Et Sums)
    m_blockLength.insert(make_pair(0x682,2));      // ConcElec: Input TrigPathC (Ht Sums)
    m_blockLength.insert(make_pair(0x683,6));      // ConcElec: EM Cands and Energy Sums Output to GT
    m_blockLength.insert(make_pair(0x686,2));      // ConcElec: Test (GT Serdes Loopback)
    m_blockLength.insert(make_pair(0x687,4));      // ConcElec: BX & Orbit Info
    // Electron Leaf FPGAs
    m_blockLength.insert(make_pair(0x800,20));     // Leaf0ElecPosEtaU1: Sort Input
    m_blockLength.insert(make_pair(0x803,4));      // Leaf0ElecPosEtaU1: Sort Output
    m_blockLength.insert(make_pair(0x804,15));     // Leaf0ElecPosEtaU1: Raw Input
    m_blockLength.insert(make_pair(0x880,16));     // Leaf0ElecPosEtaU2: Sort Input
    m_blockLength.insert(make_pair(0x883,4));      // Leaf0ElecPosEtaU2: Sort Output
    m_blockLength.insert(make_pair(0x884,12));     // Leaf0ElecPosEtaU2: Raw Input
    m_blockLength.insert(make_pair(0xc00,20));     // Leaf0ElecNegEtaU1: Sort Input
    m_blockLength.insert(make_pair(0xc03,4));      // Leaf0ElecNegEtaU1: Sort Output
    m_blockLength.insert(make_pair(0xc04,15));     // Leaf0ElecNegEtaU1: Raw Input
    m_blockLength.insert(make_pair(0xc80,16));     // Leaf0ElecNegEtaU2: Sort Input
    m_blockLength.insert(make_pair(0xc83,4));      // Leaf0ElecNegEtaU2: Sort Output
    m_blockLength.insert(make_pair(0xc84,12));     // Leaf0ElecNegEtaU2: Raw Input
    // Wheel Pos-eta Jet FPGA
    m_blockLength.insert(make_pair(0x300,27));     // WheelPosEtaJet: Input TrigPathA (Jet Sort)
    m_blockLength.insert(make_pair(0x303,6));      // WheelPosEtaJet: Output TrigPathA (Jet Sort)
    m_blockLength.insert(make_pair(0x306,32));     // WheelPosEtaJet: Test (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    m_blockLength.insert(make_pair(0x307,4));      // WheelPosEtaJet: Info (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    // Wheel Pos-eta Energy FPGA
    m_blockLength.insert(make_pair(0x380,21));     // WheelPosEtaEnergy: Input TrigPathA (Et)
    m_blockLength.insert(make_pair(0x381,3));      // WheelPosEtaEnergy: Input TrigPathB (Ht)
    m_blockLength.insert(make_pair(0x383,7));      // WheelPosEtaEnergy: Output TrigPathA (Et)
    m_blockLength.insert(make_pair(0x385,2));      // WheelPosEtaEnergy: Output TrigPathB (Ht)
    m_blockLength.insert(make_pair(0x386,32));     // WheelPosEtaEnergy: Test
    m_blockLength.insert(make_pair(0x387,6));      // WheelPosEtaEnergy: BX & Orbit Info   (Potential data incompatibility between V24/V25 where block length=4, and V27.1 where block length=6)
    // Wheel Neg-eta Jet FPGA
    m_blockLength.insert(make_pair(0x700,27));     // WheelNegEtaJet: Input TrigPathA (Jet Sort)
    m_blockLength.insert(make_pair(0x703,6));      // WheelNegEtaJet: Output TrigPathA (Jet Sort)
    m_blockLength.insert(make_pair(0x706,32));     // WheelNegEtaJet: Test (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    m_blockLength.insert(make_pair(0x707,4));      // WheelNegEtaJet: Info (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    // Wheel Neg-eta Energy FPGA
    m_blockLength.insert(make_pair(0x780,21));     // WheelNegEtaEnergy: Input TrigPathA (Et)
    m_blockLength.insert(make_pair(0x781,3));      // WheelNegEtaEnergy: Input TrigPathB (Ht)
    m_blockLength.insert(make_pair(0x783,7));      // WheelNegEtaEnergy: Output TrigPathA (Et)
    m_blockLength.insert(make_pair(0x785,2));      // WheelNegEtaEnergy: Output TrigPathB (Ht)
    m_blockLength.insert(make_pair(0x786,32));     // WheelNegEtaEnergy: Test
    m_blockLength.insert(make_pair(0x787,6));      // WheelNegEtaEnergy: BX & Orbit Info   (Potential data incompatibility between V24/V25 where block length=4, and V27.1 where block length=6)
    // Jet Leaf FPGAs - Positive Eta 
    m_blockLength.insert(make_pair(0x900,12));     // Leaf1JetPosEtaU1: JF2 Input
    m_blockLength.insert(make_pair(0x901,3));      // Leaf1JetPosEtaU1: JF2 Shared Received
    m_blockLength.insert(make_pair(0x902,3));      // Leaf1JetPosEtaU1: JF2 Shared Sent
    m_blockLength.insert(make_pair(0x903,10));     // Leaf1JetPosEtaU1: JF2 Output
    m_blockLength.insert(make_pair(0x904,8));      // Leaf1JetPosEtaU1: JF2 Raw Input
    m_blockLength.insert(make_pair(0x908,12));     // Leaf1JetPosEtaU1: JF3 Input
    m_blockLength.insert(make_pair(0x909,3));      // Leaf1JetPosEtaU1: JF3 Shared Received
    m_blockLength.insert(make_pair(0x90a,3));      // Leaf1JetPosEtaU1: JF3 Shared Sent
    m_blockLength.insert(make_pair(0x90b,10));     // Leaf1JetPosEtaU1: JF3 Output
    m_blockLength.insert(make_pair(0x90c,8));      // Leaf1JetPosEtaU1: JF3 Raw Input
    m_blockLength.insert(make_pair(0x980,3));      // Leaf1JetPosEtaU2: Eta0 Input
    m_blockLength.insert(make_pair(0x984,6));      // Leaf1JetPosEtaU2: Eta0 Raw Input
    m_blockLength.insert(make_pair(0x988,12));     // Leaf1JetPosEtaU2: JF1 Input
    m_blockLength.insert(make_pair(0x989,3));      // Leaf1JetPosEtaU2: JF1 Shared Received
    m_blockLength.insert(make_pair(0x98a,3));      // Leaf1JetPosEtaU2: JF1 Shared Sent
    m_blockLength.insert(make_pair(0x98b,10));     // Leaf1JetPosEtaU2: JF1 Output
    m_blockLength.insert(make_pair(0x98c,8));      // Leaf1JetPosEtaU2: JF1 Raw Input
    m_blockLength.insert(make_pair(0xa00,12));     // Leaf2JetPosEtaU1: JF2 Input
    m_blockLength.insert(make_pair(0xa01,3));      // Leaf2JetPosEtaU1: JF2 Shared Received
    m_blockLength.insert(make_pair(0xa02,3));      // Leaf2JetPosEtaU1: JF2 Shared Sent
    m_blockLength.insert(make_pair(0xa03,10));     // Leaf2JetPosEtaU1: JF2 Output
    m_blockLength.insert(make_pair(0xa04,8));      // Leaf2JetPosEtaU1: JF2 Raw Input
    m_blockLength.insert(make_pair(0xa08,12));     // Leaf2JetPosEtaU1: JF3 Input
    m_blockLength.insert(make_pair(0xa09,3));      // Leaf2JetPosEtaU1: JF3 Shared Received
    m_blockLength.insert(make_pair(0xa0a,3));      // Leaf2JetPosEtaU1: JF3 Shared Sent
    m_blockLength.insert(make_pair(0xa0b,10));     // Leaf2JetPosEtaU1: JF3 Output
    m_blockLength.insert(make_pair(0xa0c,8));      // Leaf2JetPosEtaU1: JF3 Raw Input
    m_blockLength.insert(make_pair(0xa80,3));      // Leaf2JetPosEtaU2: Eta0 Input
    m_blockLength.insert(make_pair(0xa84,6));      // Leaf2JetPosEtaU2: Eta0 Raw Input
    m_blockLength.insert(make_pair(0xa88,12));     // Leaf2JetPosEtaU2: JF1 Input
    m_blockLength.insert(make_pair(0xa89,3));      // Leaf2JetPosEtaU2: JF1 Shared Received
    m_blockLength.insert(make_pair(0xa8a,3));      // Leaf2JetPosEtaU2: JF1 Shared Sent
    m_blockLength.insert(make_pair(0xa8b,10));     // Leaf2JetPosEtaU2: JF1 Output
    m_blockLength.insert(make_pair(0xa8c,8));      // Leaf2JetPosEtaU2: JF1 Raw Input
    m_blockLength.insert(make_pair(0xb00,12));     // Leaf3JetPosEtaU1: JF2 Input
    m_blockLength.insert(make_pair(0xb01,3));      // Leaf3JetPosEtaU1: JF2 Shared Received
    m_blockLength.insert(make_pair(0xb02,3));      // Leaf3JetPosEtaU1: JF2 Shared Sent
    m_blockLength.insert(make_pair(0xb03,10));     // Leaf3JetPosEtaU1: JF2 Output
    m_blockLength.insert(make_pair(0xb04,8));      // Leaf3JetPosEtaU1: JF2 Raw Input
    m_blockLength.insert(make_pair(0xb08,12));     // Leaf3JetPosEtaU1: JF3 Input
    m_blockLength.insert(make_pair(0xb09,3));      // Leaf3JetPosEtaU1: JF3 Shared Received
    m_blockLength.insert(make_pair(0xb0a,3));      // Leaf3JetPosEtaU1: JF3 Shared Sent
    m_blockLength.insert(make_pair(0xb0b,10));     // Leaf3JetPosEtaU1: JF3 Output
    m_blockLength.insert(make_pair(0xb0c,8));      // Leaf3JetPosEtaU1: JF3 Raw Input
    m_blockLength.insert(make_pair(0xb80,3));      // Leaf3JetPosEtaU2: Eta0 Input
    m_blockLength.insert(make_pair(0xb84,6));      // Leaf3JetPosEtaU2: Eta0 Raw Input
    m_blockLength.insert(make_pair(0xb88,12));     // Leaf3JetPosEtaU2: JF1 Input
    m_blockLength.insert(make_pair(0xb89,3));      // Leaf3JetPosEtaU2: JF1 Shared Received
    m_blockLength.insert(make_pair(0xb8a,3));      // Leaf3JetPosEtaU2: JF1 Shared Sent
    m_blockLength.insert(make_pair(0xb8b,10));     // Leaf3JetPosEtaU2: JF1 Output
    m_blockLength.insert(make_pair(0xb8c,8));      // Leaf3JetPosEtaU2: JF1 Raw Input
    // Jet Leaf FPGAs - Negative Eta 
    m_blockLength.insert(make_pair(0xd00,12));     // Leaf1JetNegEtaU1: JF2 Input
    m_blockLength.insert(make_pair(0xd01,3));      // Leaf1JetNegEtaU1: JF2 Shared Received
    m_blockLength.insert(make_pair(0xd02,3));      // Leaf1JetNegEtaU1: JF2 Shared Sent
    m_blockLength.insert(make_pair(0xd03,10));     // Leaf1JetNegEtaU1: JF2 Output
    m_blockLength.insert(make_pair(0xd04,8));      // Leaf1JetNegEtaU1: JF2 Raw Input
    m_blockLength.insert(make_pair(0xd08,12));     // Leaf1JetNegEtaU1: JF3 Input
    m_blockLength.insert(make_pair(0xd09,3));      // Leaf1JetNegEtaU1: JF3 Shared Received
    m_blockLength.insert(make_pair(0xd0a,3));      // Leaf1JetNegEtaU1: JF3 Shared Sent
    m_blockLength.insert(make_pair(0xd0b,10));     // Leaf1JetNegEtaU1: JF3 Output
    m_blockLength.insert(make_pair(0xd0c,8));      // Leaf1JetNegEtaU1: JF3 Raw Input
    m_blockLength.insert(make_pair(0xd80,3));      // Leaf1JetNegEtaU2: Eta0 Input
    m_blockLength.insert(make_pair(0xd84,6));      // Leaf1JetNegEtaU2: Eta0 Raw Input
    m_blockLength.insert(make_pair(0xd88,12));     // Leaf1JetNegEtaU2: JF1 Input
    m_blockLength.insert(make_pair(0xd89,3));      // Leaf1JetNegEtaU2: JF1 Shared Received
    m_blockLength.insert(make_pair(0xd8a,3));      // Leaf1JetNegEtaU2: JF1 Shared Sent
    m_blockLength.insert(make_pair(0xd8b,10));     // Leaf1JetNegEtaU2: JF1 Output
    m_blockLength.insert(make_pair(0xd8c,8));      // Leaf1JetNegEtaU2: JF1 Raw Input
    m_blockLength.insert(make_pair(0xe00,12));     // Leaf2JetNegEtaU1: JF2 Input
    m_blockLength.insert(make_pair(0xe01,3));      // Leaf2JetNegEtaU1: JF2 Shared Received
    m_blockLength.insert(make_pair(0xe02,3));      // Leaf2JetNegEtaU1: JF2 Shared Sent
    m_blockLength.insert(make_pair(0xe03,10));     // Leaf2JetNegEtaU1: JF2 Output
    m_blockLength.insert(make_pair(0xe04,8));      // Leaf2JetNegEtaU1: JF2 Raw Input
    m_blockLength.insert(make_pair(0xe08,12));     // Leaf2JetNegEtaU1: JF3 Input
    m_blockLength.insert(make_pair(0xe09,3));      // Leaf2JetNegEtaU1: JF3 Shared Received
    m_blockLength.insert(make_pair(0xe0a,3));      // Leaf2JetNegEtaU1: JF3 Shared Sent
    m_blockLength.insert(make_pair(0xe0b,10));     // Leaf2JetNegEtaU1: JF3 Output
    m_blockLength.insert(make_pair(0xe0c,8));      // Leaf2JetNegEtaU1: JF3 Raw Input
    m_blockLength.insert(make_pair(0xe80,3));      // Leaf2JetNegEtaU2: Eta0 Input
    m_blockLength.insert(make_pair(0xe84,6));      // Leaf2JetNegEtaU2: Eta0 Raw Input
    m_blockLength.insert(make_pair(0xe88,12));     // Leaf2JetNegEtaU2: JF1 Input
    m_blockLength.insert(make_pair(0xe89,3));      // Leaf2JetNegEtaU2: JF1 Shared Received
    m_blockLength.insert(make_pair(0xe8a,3));      // Leaf2JetNegEtaU2: JF1 Shared Sent
    m_blockLength.insert(make_pair(0xe8b,10));     // Leaf2JetNegEtaU2: JF1 Output
    m_blockLength.insert(make_pair(0xe8c,8));      // Leaf2JetNegEtaU2: JF1 Raw Input
    m_blockLength.insert(make_pair(0xf00,12));     // Leaf3JetNegEtaU1: JF2 Input
    m_blockLength.insert(make_pair(0xf01,3));      // Leaf3JetNegEtaU1: JF2 Shared Received
    m_blockLength.insert(make_pair(0xf02,3));      // Leaf3JetNegEtaU1: JF2 Shared Sent
    m_blockLength.insert(make_pair(0xf03,10));     // Leaf3JetNegEtaU1: JF2 Output
    m_blockLength.insert(make_pair(0xf04,8));      // Leaf3JetNegEtaU1: JF2 Raw Input
    m_blockLength.insert(make_pair(0xf08,12));     // Leaf3JetNegEtaU1: JF3 Input
    m_blockLength.insert(make_pair(0xf09,3));      // Leaf3JetNegEtaU1: JF3 Shared Received
    m_blockLength.insert(make_pair(0xf0a,3));      // Leaf3JetNegEtaU1: JF3 Shared Sent
    m_blockLength.insert(make_pair(0xf0b,10));     // Leaf3JetNegEtaU1: JF3 Output
    m_blockLength.insert(make_pair(0xf0c,8));      // Leaf3JetNegEtaU1: JF3 Raw Input
    m_blockLength.insert(make_pair(0xf80,3));      // Leaf3JetNegEtaU2: Eta0 Input
    m_blockLength.insert(make_pair(0xf84,6));      // Leaf3JetNegEtaU2: Eta0 Raw Input
    m_blockLength.insert(make_pair(0xf88,12));     // Leaf3JetNegEtaU2: JF1 Input    
    m_blockLength.insert(make_pair(0xf89,3));      // Leaf3JetNegEtaU2: JF1 Shared Received
    m_blockLength.insert(make_pair(0xf8a,3));      // Leaf3JetNegEtaU2: JF1 Shared Sent
    m_blockLength.insert(make_pair(0xf8b,10));     // Leaf3JetNegEtaU2: JF1 Output
    m_blockLength.insert(make_pair(0xf8c,8));      // Leaf3JetNegEtaU2: JF1 Raw Input


    /*** Setup BlockID to BlockName Map ***/
    // Miscellaneous Blocks
    m_blockName.insert(make_pair(0x000,"NULL"));
    // ConcJet FPGA
    m_blockName.insert(make_pair(0x580,"ConcJet: Input TrigPathA (Jet Cands)"));
    m_blockName.insert(make_pair(0x581,"ConcJet: Input TrigPathB (HF Rings)"));
    m_blockName.insert(make_pair(0x583,"ConcJet: Jet Cands and Counts Output to GT"));
    m_blockName.insert(make_pair(0x587,"ConcJet: BX & Orbit Info"));
    // ConcElec FPGA
    m_blockName.insert(make_pair(0x680,"ConcElec: Input TrigPathA (EM Cands)"));
    m_blockName.insert(make_pair(0x681,"ConcElec: Input TrigPathB (Et Sums)"));
    m_blockName.insert(make_pair(0x682,"ConcElec: Input TrigPathC (Ht Sums)"));
    m_blockName.insert(make_pair(0x683,"ConcElec: EM Cands and Energy Sums Output to GT"));
    m_blockName.insert(make_pair(0x686,"ConcElec: Test (GT Serdes Loopback)"));
    m_blockName.insert(make_pair(0x687,"ConcElec: BX & Orbit Info"));
    // Electron Leaf FPGAs
    m_blockName.insert(make_pair(0x800,"Leaf0ElecPosEtaU1: Sort Input"));
    m_blockName.insert(make_pair(0x803,"Leaf0ElecPosEtaU1: Sort Output"));
    m_blockName.insert(make_pair(0x804,"Leaf0ElecPosEtaU1: Raw Input"));
    m_blockName.insert(make_pair(0x880,"Leaf0ElecPosEtaU2: Sort Input"));
    m_blockName.insert(make_pair(0x883,"Leaf0ElecPosEtaU2: Sort Output"));
    m_blockName.insert(make_pair(0x884,"Leaf0ElecPosEtaU2: Raw Input"));
    m_blockName.insert(make_pair(0xc00,"Leaf0ElecNegEtaU1: Sort Input"));
    m_blockName.insert(make_pair(0xc03,"Leaf0ElecNegEtaU1: Sort Output"));
    m_blockName.insert(make_pair(0xc04,"Leaf0ElecNegEtaU1: Raw Input"));
    m_blockName.insert(make_pair(0xc80,"Leaf0ElecNegEtaU2: Sort Input"));
    m_blockName.insert(make_pair(0xc83,"Leaf0ElecNegEtaU2: Sort Output"));
    m_blockName.insert(make_pair(0xc84,"Leaf0ElecNegEtaU2: Raw Input"));
    // Wheel Pos-eta Jet FPGA
    m_blockName.insert(make_pair(0x300,"WheelPosEtaJet: Input TrigPathA (Jet Sort)"));
    m_blockName.insert(make_pair(0x303,"WheelPosEtaJet: Output TrigPathA (Jet Sort)"));
    m_blockName.insert(make_pair(0x306,"WheelPosEtaJet: Test (deprecated)"));  // (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    m_blockName.insert(make_pair(0x307,"WheelPosEtaJet: Info (deprecated)"));  // (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    // Wheel Pos-eta Energy FPGA
    m_blockName.insert(make_pair(0x380,"WheelPosEtaEnergy: Input TrigPathA (Et)"));
    m_blockName.insert(make_pair(0x381,"WheelPosEtaEnergy: Input TrigPathB (Ht)"));
    m_blockName.insert(make_pair(0x383,"WheelPosEtaEnergy: Output TrigPathA (Et)"));
    m_blockName.insert(make_pair(0x385,"WheelPosEtaEnergy: Output TrigPathB (Ht)"));
    m_blockName.insert(make_pair(0x386,"WheelPosEtaEnergy: Test"));
    m_blockName.insert(make_pair(0x387,"WheelPosEtaEnergy: BX & Orbit Info"));
    // Wheel Neg-eta Jet FPGA
    m_blockName.insert(make_pair(0x700,"WheelNegEtaJet: Input TrigPathA (Jet Sort)"));
    m_blockName.insert(make_pair(0x703,"WheelNegEtaJet: Output TrigPathA (Jet Sort)"));
    m_blockName.insert(make_pair(0x706,"WheelNegEtaJet: Test (deprecated)"));  // (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    m_blockName.insert(make_pair(0x707,"WheelNegEtaJet: Info (deprecated)"));  // (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    // Wheel Neg-eta Energy FPGA
    m_blockName.insert(make_pair(0x780,"WheelNegEtaEnergy: Input TrigPathA (Et)"));
    m_blockName.insert(make_pair(0x781,"WheelNegEtaEnergy: Input TrigPathB (Ht)"));
    m_blockName.insert(make_pair(0x783,"WheelNegEtaEnergy: Output TrigPathA (Et)"));
    m_blockName.insert(make_pair(0x785,"WheelNegEtaEnergy: Output TrigPathB (Ht)"));
    m_blockName.insert(make_pair(0x786,"WheelNegEtaEnergy: Test"));
    m_blockName.insert(make_pair(0x787,"WheelNegEtaEnergy: BX & Orbit Info"));
    // Jet Leaf FPGAs - Positive Eta
    m_blockName.insert(make_pair(0x900,"Leaf1JetPosEtaU1: JF2 Input"));
    m_blockName.insert(make_pair(0x901,"Leaf1JetPosEtaU1: JF2 Shared Received"));
    m_blockName.insert(make_pair(0x902,"Leaf1JetPosEtaU1: JF2 Shared Sent"));
    m_blockName.insert(make_pair(0x903,"Leaf1JetPosEtaU1: JF2 Output"));
    m_blockName.insert(make_pair(0x904,"Leaf1JetPosEtaU1: JF2 Raw Input"));
    m_blockName.insert(make_pair(0x908,"Leaf1JetPosEtaU1: JF3 Input"));
    m_blockName.insert(make_pair(0x909,"Leaf1JetPosEtaU1: JF3 Shared Received"));
    m_blockName.insert(make_pair(0x90a,"Leaf1JetPosEtaU1: JF3 Shared Sent"));
    m_blockName.insert(make_pair(0x90b,"Leaf1JetPosEtaU1: JF3 Output"));
    m_blockName.insert(make_pair(0x90c,"Leaf1JetPosEtaU1: JF3 Raw Input"));
    m_blockName.insert(make_pair(0x980,"Leaf1JetPosEtaU2: Eta0 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0x984,"Leaf1JetPosEtaU2: Eta0 Raw Input"));
    m_blockName.insert(make_pair(0x988,"Leaf1JetPosEtaU2: JF1 Input"));
    m_blockName.insert(make_pair(0x989,"Leaf1JetPosEtaU2: JF1 Shared Received"));
    m_blockName.insert(make_pair(0x98a,"Leaf1JetPosEtaU2: JF1 Shared Sent"));
    m_blockName.insert(make_pair(0x98b,"Leaf1JetPosEtaU2: JF1 Output"));
    m_blockName.insert(make_pair(0x98c,"Leaf1JetPosEtaU2: JF1 Raw Input"));
    m_blockName.insert(make_pair(0xa00,"Leaf2JetPosEtaU1: JF2 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xa01,"Leaf2JetPosEtaU1: JF2 Shared Received"));
    m_blockName.insert(make_pair(0xa02,"Leaf2JetPosEtaU1: JF2 Shared Sent"));
    m_blockName.insert(make_pair(0xa03,"Leaf2JetPosEtaU1: JF2 Output"));
    m_blockName.insert(make_pair(0xa04,"Leaf2JetPosEtaU1: JF2 Raw Input"));
    m_blockName.insert(make_pair(0xa08,"Leaf2JetPosEtaU1: JF3 Input"));
    m_blockName.insert(make_pair(0xa09,"Leaf2JetPosEtaU1: JF3 Shared Received"));
    m_blockName.insert(make_pair(0xa0a,"Leaf2JetPosEtaU1: JF3 Shared Sent"));
    m_blockName.insert(make_pair(0xa0b,"Leaf2JetPosEtaU1: JF3 Output"));
    m_blockName.insert(make_pair(0xa0c,"Leaf2JetPosEtaU1: JF3 Raw Input"));
    m_blockName.insert(make_pair(0xa80,"Leaf2JetPosEtaU2: Eta0 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xa84,"Leaf2JetPosEtaU2: Eta0 Raw Input"));
    m_blockName.insert(make_pair(0xa88,"Leaf2JetPosEtaU2: JF1 Input"));
    m_blockName.insert(make_pair(0xa89,"Leaf2JetPosEtaU2: JF1 Shared Received"));
    m_blockName.insert(make_pair(0xa8a,"Leaf2JetPosEtaU2: JF1 Shared Sent"));
    m_blockName.insert(make_pair(0xa8b,"Leaf2JetPosEtaU2: JF1 Output"));
    m_blockName.insert(make_pair(0xa8c,"Leaf2JetPosEtaU2: JF1 Raw Input"));
    m_blockName.insert(make_pair(0xb00,"Leaf3JetPosEtaU1: JF2 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xb01,"Leaf3JetPosEtaU1: JF2 Shared Received"));
    m_blockName.insert(make_pair(0xb02,"Leaf3JetPosEtaU1: JF2 Shared Sent"));
    m_blockName.insert(make_pair(0xb03,"Leaf3JetPosEtaU1: JF2 Output"));
    m_blockName.insert(make_pair(0xb04,"Leaf3JetPosEtaU1: JF2 Raw Input"));
    m_blockName.insert(make_pair(0xb08,"Leaf3JetPosEtaU1: JF3 Input"));
    m_blockName.insert(make_pair(0xb09,"Leaf3JetPosEtaU1: JF3 Shared Received"));
    m_blockName.insert(make_pair(0xb0a,"Leaf3JetPosEtaU1: JF3 Shared Sent"));
    m_blockName.insert(make_pair(0xb0b,"Leaf3JetPosEtaU1: JF3 Output"));
    m_blockName.insert(make_pair(0xb0c,"Leaf3JetPosEtaU1: JF3 Raw Input"));
    m_blockName.insert(make_pair(0xb80,"Leaf3JetPosEtaU2: Eta0 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xb84,"Leaf3JetPosEtaU2: Eta0 Raw Input"));
    m_blockName.insert(make_pair(0xb88,"Leaf3JetPosEtaU2: JF1 Input"));
    m_blockName.insert(make_pair(0xb89,"Leaf3JetPosEtaU2: JF1 Shared Received"));
    m_blockName.insert(make_pair(0xb8a,"Leaf3JetPosEtaU2: JF1 Shared Sent"));
    m_blockName.insert(make_pair(0xb8b,"Leaf3JetPosEtaU2: JF1 Output"));
    m_blockName.insert(make_pair(0xb8c,"Leaf3JetPosEtaU2: JF1 Raw Input"));
    // Jet Leaf FPGAs - Negative Eta
    m_blockName.insert(make_pair(0xd00,"Leaf1JetNegEtaU1: JF2 Input"));       // START OF NEG ETA JET LEAVES
    m_blockName.insert(make_pair(0xd01,"Leaf1JetNegEtaU1: JF2 Shared Received"));
    m_blockName.insert(make_pair(0xd02,"Leaf1JetNegEtaU1: JF2 Shared Sent"));
    m_blockName.insert(make_pair(0xd03,"Leaf1JetNegEtaU1: JF2 Output"));
    m_blockName.insert(make_pair(0xd04,"Leaf1JetNegEtaU1: JF2 Raw Input"));
    m_blockName.insert(make_pair(0xd08,"Leaf1JetNegEtaU1: JF3 Input"));
    m_blockName.insert(make_pair(0xd09,"Leaf1JetNegEtaU1: JF3 Shared Received"));
    m_blockName.insert(make_pair(0xd0a,"Leaf1JetNegEtaU1: JF3 Shared Sent"));
    m_blockName.insert(make_pair(0xd0b,"Leaf1JetNegEtaU1: JF3 Output"));
    m_blockName.insert(make_pair(0xd0c,"Leaf1JetNegEtaU1: JF3 Raw Input"));
    m_blockName.insert(make_pair(0xd80,"Leaf1JetNegEtaU2: Eta0 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xd84,"Leaf1JetNegEtaU2: Eta0 Raw Input"));
    m_blockName.insert(make_pair(0xd88,"Leaf1JetNegEtaU2: JF1 Input"));
    m_blockName.insert(make_pair(0xd89,"Leaf1JetNegEtaU2: JF1 Shared Received"));
    m_blockName.insert(make_pair(0xd8a,"Leaf1JetNegEtaU2: JF1 Shared Sent"));
    m_blockName.insert(make_pair(0xd8b,"Leaf1JetNegEtaU2: JF1 Output"));
    m_blockName.insert(make_pair(0xd8c,"Leaf1JetNegEtaU2: JF1 Raw Input"));
    m_blockName.insert(make_pair(0xe00,"Leaf2JetNegEtaU1: JF2 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xe01,"Leaf2JetNegEtaU1: JF2 Shared Received"));
    m_blockName.insert(make_pair(0xe02,"Leaf2JetNegEtaU1: JF2 Shared Sent"));
    m_blockName.insert(make_pair(0xe03,"Leaf2JetNegEtaU1: JF2 Output"));
    m_blockName.insert(make_pair(0xe04,"Leaf2JetNegEtaU1: JF2 Raw Input"));
    m_blockName.insert(make_pair(0xe08,"Leaf2JetNegEtaU1: JF3 Input"));
    m_blockName.insert(make_pair(0xe09,"Leaf2JetNegEtaU1: JF3 Shared Received"));
    m_blockName.insert(make_pair(0xe0a,"Leaf2JetNegEtaU1: JF3 Shared Sent"));
    m_blockName.insert(make_pair(0xe0b,"Leaf2JetNegEtaU1: JF3 Output"));
    m_blockName.insert(make_pair(0xe0c,"Leaf2JetNegEtaU1: JF3 Raw Input"));
    m_blockName.insert(make_pair(0xe80,"Leaf2JetNegEtaU2: Eta0 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xe84,"Leaf2JetNegEtaU2: Eta0 Raw Input"));
    m_blockName.insert(make_pair(0xe88,"Leaf2JetNegEtaU2: JF1 Input"));
    m_blockName.insert(make_pair(0xe89,"Leaf2JetNegEtaU2: JF1 Shared Received"));
    m_blockName.insert(make_pair(0xe8a,"Leaf2JetNegEtaU2: JF1 Shared Sent"));
    m_blockName.insert(make_pair(0xe8b,"Leaf2JetNegEtaU2: JF1 Output"));
    m_blockName.insert(make_pair(0xe8c,"Leaf2JetNegEtaU2: JF1 Raw Input"));
    m_blockName.insert(make_pair(0xf00,"Leaf3JetNegEtaU1: JF2 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xf01,"Leaf3JetNegEtaU1: JF2 Shared Received"));
    m_blockName.insert(make_pair(0xf02,"Leaf3JetNegEtaU1: JF2 Shared Sent"));
    m_blockName.insert(make_pair(0xf03,"Leaf3JetNegEtaU1: JF2 Output"));
    m_blockName.insert(make_pair(0xf04,"Leaf3JetNegEtaU1: JF2 Raw Input"));
    m_blockName.insert(make_pair(0xf08,"Leaf3JetNegEtaU1: JF3 Input"));
    m_blockName.insert(make_pair(0xf09,"Leaf3JetNegEtaU1: JF3 Shared Received"));
    m_blockName.insert(make_pair(0xf0a,"Leaf3JetNegEtaU1: JF3 Shared Sent"));
    m_blockName.insert(make_pair(0xf0b,"Leaf3JetNegEtaU1: JF3 Output"));
    m_blockName.insert(make_pair(0xf0c,"Leaf3JetNegEtaU1: JF3 Raw Input"));
    m_blockName.insert(make_pair(0xf80,"Leaf3JetNegEtaU2: Eta0 Input"));  // Next Leaf Start
    m_blockName.insert(make_pair(0xf84,"Leaf3JetNegEtaU2: Eta0 Raw Input"));
    m_blockName.insert(make_pair(0xf88,"Leaf3JetNegEtaU2: JF1 Input"));
    m_blockName.insert(make_pair(0xf89,"Leaf3JetNegEtaU2: JF1 Shared Received"));
    m_blockName.insert(make_pair(0xf8a,"Leaf3JetNegEtaU2: JF1 Shared Sent"));
    m_blockName.insert(make_pair(0xf8b,"Leaf3JetNegEtaU2: JF1 Output"));
    m_blockName.insert(make_pair(0xf8c,"Leaf3JetNegEtaU2: JF1 Raw Input"));


    /*** Setup BlockID to Unpack-Function Map ***/
    // Miscellaneous Blocks
    m_blockUnpackFn[0x000] = &GctFormatTranslateV35::blockDoNothing;                    // NULL
    // ConcJet FPGA                                                             
    m_blockUnpackFn[0x580] = &GctFormatTranslateV35::blockToGctTrigObjects;             // ConcJet: Input TrigPathA (Jet Cands)
    m_blockUnpackFn[0x581] = &GctFormatTranslateV35::blockToGctInternRingSums;          // ConcJet: Input TrigPathB (HF Rings)
    m_blockUnpackFn[0x583] = &GctFormatTranslateV35::blockToGctJetCandsAndCounts;       // ConcJet: Jet Cands and Counts Output to GT
    m_blockUnpackFn[0x587] = &GctFormatTranslateV35::blockDoNothing;                    // ConcJet: BX & Orbit Info
    // ConcElec FPGA                                                            
    m_blockUnpackFn[0x680] = &GctFormatTranslateV35::blockToGctInternEmCand;            // ConcElec: Input TrigPathA (EM Cands)
    m_blockUnpackFn[0x681] = &GctFormatTranslateV35::blockToGctInternEtSums;            // ConcElec: Input TrigPathB (Et Sums)
    m_blockUnpackFn[0x682] = &GctFormatTranslateV35::blockToGctInternEtSums;            // ConcElec: Input TrigPathC (Ht Sums)
    m_blockUnpackFn[0x683] = &GctFormatTranslateV35::blockToGctEmCandsAndEnergySums;    // ConcElec: EM Cands and Energy Sums Output to GT
    m_blockUnpackFn[0x686] = &GctFormatTranslateV35::blockDoNothing;                    // ConcElec: Test (GT Serdes Loopback)
    m_blockUnpackFn[0x687] = &GctFormatTranslateV35::blockDoNothing;                    // ConcElec: BX & Orbit Info
    // Electron Leaf FPGAs                                                      
    m_blockUnpackFn[0x800] = &GctFormatTranslateV35::blockToGctInternEmCand;            // Leaf0ElecPosEtaU1: Sort Input
    m_blockUnpackFn[0x803] = &GctFormatTranslateV35::blockToGctInternEmCand;            // Leaf0ElecPosEtaU1: Sort Output
    m_blockUnpackFn[0x804] = &GctFormatTranslateV35::blockToFibresAndToRctEmCand;       // Leaf0ElecPosEtaU1: Raw Input
    m_blockUnpackFn[0x880] = &GctFormatTranslateV35::blockToGctInternEmCand;            // Leaf0ElecPosEtaU2: Sort Input
    m_blockUnpackFn[0x883] = &GctFormatTranslateV35::blockToGctInternEmCand;            // Leaf0ElecPosEtaU2: Sort Output
    m_blockUnpackFn[0x884] = &GctFormatTranslateV35::blockToFibresAndToRctEmCand;       // Leaf0ElecPosEtaU2: Raw Input
    m_blockUnpackFn[0xc00] = &GctFormatTranslateV35::blockToGctInternEmCand;            // Leaf0ElecNegEtaU1: Sort Input
    m_blockUnpackFn[0xc03] = &GctFormatTranslateV35::blockToGctInternEmCand;            // Leaf0ElecNegEtaU1: Sort Output
    m_blockUnpackFn[0xc04] = &GctFormatTranslateV35::blockToFibresAndToRctEmCand;       // Leaf0ElecNegEtaU1: Raw Input
    m_blockUnpackFn[0xc80] = &GctFormatTranslateV35::blockToGctInternEmCand;            // Leaf0ElecNegEtaU2: Sort Input
    m_blockUnpackFn[0xc83] = &GctFormatTranslateV35::blockToGctInternEmCand;            // Leaf0ElecNegEtaU2: Sort Output
    m_blockUnpackFn[0xc84] = &GctFormatTranslateV35::blockToFibresAndToRctEmCand;       // Leaf0ElecNegEtaU2: Raw Input
    // Wheel Pos-eta Jet FPGA                                                   
    m_blockUnpackFn[0x300] = &GctFormatTranslateV35::blockToGctJetClusterMinimal;       // WheelPosEtaJet: Input TrigPathA (Jet Sort)
    m_blockUnpackFn[0x303] = &GctFormatTranslateV35::blockToGctTrigObjects;             // WheelPosEtaJet: Output TrigPathA (Jet Sort)
    m_blockUnpackFn[0x306] = &GctFormatTranslateV35::blockDoNothing;                    // WheelPosEtaJet: Test (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    m_blockUnpackFn[0x307] = &GctFormatTranslateV35::blockDoNothing;                    // WheelPosEtaJet: Info (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    // Wheel Pos-eta Energy FPGA                                                
    m_blockUnpackFn[0x380] = &GctFormatTranslateV35::blockToGctWheelInputInternEtAndRingSums;     // WheelPosEtaEnergy: Input TrigPathA (Et)
    m_blockUnpackFn[0x381] = &GctFormatTranslateV35::blockToGctInternEtSums;            // WheelPosEtaEnergy: Input TrigPathB (Ht)
    m_blockUnpackFn[0x383] = &GctFormatTranslateV35::blockToGctWheelOutputInternEtAndRingSums;     // WheelPosEtaEnergy: Output TrigPathA (Et)
    m_blockUnpackFn[0x385] = &GctFormatTranslateV35::blockToGctInternEtSums;            // WheelPosEtaEnergy: Output TrigPathB (Ht)
    m_blockUnpackFn[0x386] = &GctFormatTranslateV35::blockDoNothing;                    // WheelPosEtaEnergy: Test
    m_blockUnpackFn[0x387] = &GctFormatTranslateV35::blockDoNothing;                    // WheelPosEtaEnergy: BX & Orbit Info   (Potential data incompatibility between V24/V25 where block length=4, and V27.1 where block length=6)
    // Wheel Neg-eta Jet FPGA                                                   
    m_blockUnpackFn[0x700] = &GctFormatTranslateV35::blockToGctJetClusterMinimal;       // WheelNegEtaJet: Input TrigPathA (Jet Sort)
    m_blockUnpackFn[0x703] = &GctFormatTranslateV35::blockToGctTrigObjects;             // WheelNegEtaJet: Output TrigPathA (Jet Sort)
    m_blockUnpackFn[0x706] = &GctFormatTranslateV35::blockDoNothing;                    // WheelNegEtaJet: Test (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    m_blockUnpackFn[0x707] = &GctFormatTranslateV35::blockDoNothing;                    // WheelNegEtaJet: Info (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    // Wheel Neg-eta Energy FPGA                                                
    m_blockUnpackFn[0x780] = &GctFormatTranslateV35::blockToGctWheelInputInternEtAndRingSums;     // WheelNegEtaEnergy: Input TrigPathA (Et)
    m_blockUnpackFn[0x781] = &GctFormatTranslateV35::blockToGctInternEtSums;            // WheelNegEtaEnergy: Input TrigPathB (Ht)
    m_blockUnpackFn[0x783] = &GctFormatTranslateV35::blockToGctWheelOutputInternEtAndRingSums;     // WheelNegEtaEnergy: Output TrigPathA (Et)
    m_blockUnpackFn[0x785] = &GctFormatTranslateV35::blockToGctInternEtSums;            // WheelNegEtaEnergy: Output TrigPathB (Ht)
    m_blockUnpackFn[0x786] = &GctFormatTranslateV35::blockDoNothing;                    // WheelNegEtaEnergy: Test
    m_blockUnpackFn[0x787] = &GctFormatTranslateV35::blockDoNothing;                    // WheelNegEtaEnergy: BX & Orbit Info   (Potential data incompatibility between V24/V25 where block length=4, and V27.1 where block length=6)
    // Jet Leaf FPGAs - Positive Eta
    m_blockUnpackFn[0x900] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf1JetPosEtaU1: JF2 Input
    m_blockUnpackFn[0x901] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetPosEtaU1: JF2 Shared Received
    m_blockUnpackFn[0x902] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetPosEtaU1: JF2 Shared Sent
    m_blockUnpackFn[0x903] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf1JetPosEtaU1: JF2 Output
    m_blockUnpackFn[0x904] = &GctFormatTranslateV35::blockToFibres;                     // Leaf1JetPosEtaU1: JF2 Raw Input
    m_blockUnpackFn[0x908] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf1JetPosEtaU1: JF3 Input
    m_blockUnpackFn[0x909] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetPosEtaU1: JF3 Shared Received
    m_blockUnpackFn[0x90a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetPosEtaU1: JF3 Shared Sent
    m_blockUnpackFn[0x90b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf1JetPosEtaU1: JF3 Output
    m_blockUnpackFn[0x90c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf1JetPosEtaU1: JF3 Raw Input
    m_blockUnpackFn[0x980] = &GctFormatTranslateV35::blockDoNothing;                    // Leaf1JetPosEtaU2: Eta0 Input
    m_blockUnpackFn[0x984] = &GctFormatTranslateV35::blockToFibres;                     // Leaf1JetPosEtaU2: Eta0 Raw Input
    m_blockUnpackFn[0x988] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf1JetPosEtaU2: JF1 Input
    m_blockUnpackFn[0x989] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetPosEtaU2: JF1 Shared Received
    m_blockUnpackFn[0x98a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetPosEtaU2: JF1 Shared Sent
    m_blockUnpackFn[0x98b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf1JetPosEtaU2: JF1 Output
    m_blockUnpackFn[0x98c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf1JetPosEtaU2: JF1 Raw Input
    m_blockUnpackFn[0xa00] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf2JetPosEtaU1: JF2 Input
    m_blockUnpackFn[0xa01] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetPosEtaU1: JF2 Shared Received
    m_blockUnpackFn[0xa02] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetPosEtaU1: JF2 Shared Sent
    m_blockUnpackFn[0xa03] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf2JetPosEtaU1: JF2 Output
    m_blockUnpackFn[0xa04] = &GctFormatTranslateV35::blockToFibres;                     // Leaf2JetPosEtaU1: JF2 Raw Input
    m_blockUnpackFn[0xa08] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf2JetPosEtaU1: JF3 Input
    m_blockUnpackFn[0xa09] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetPosEtaU1: JF3 Shared Received
    m_blockUnpackFn[0xa0a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetPosEtaU1: JF3 Shared Sent
    m_blockUnpackFn[0xa0b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf2JetPosEtaU1: JF3 Output
    m_blockUnpackFn[0xa0c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf2JetPosEtaU1: JF3 Raw Input
    m_blockUnpackFn[0xa80] = &GctFormatTranslateV35::blockDoNothing;                    // Leaf2JetPosEtaU2: Eta0 Input
    m_blockUnpackFn[0xa84] = &GctFormatTranslateV35::blockToFibres;                     // Leaf2JetPosEtaU2: Eta0 Raw Input
    m_blockUnpackFn[0xa88] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf2JetPosEtaU2: JF1 Input
    m_blockUnpackFn[0xa89] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetPosEtaU2: JF1 Shared Received
    m_blockUnpackFn[0xa8a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetPosEtaU2: JF1 Shared Sent
    m_blockUnpackFn[0xa8b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf2JetPosEtaU2: JF1 Output
    m_blockUnpackFn[0xa8c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf2JetPosEtaU2: JF1 Raw Input
    m_blockUnpackFn[0xb00] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf3JetPosEtaU1: JF2 Input
    m_blockUnpackFn[0xb01] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetPosEtaU1: JF2 Shared Received
    m_blockUnpackFn[0xb02] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetPosEtaU1: JF2 Shared Sent
    m_blockUnpackFn[0xb03] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf3JetPosEtaU1: JF2 Output
    m_blockUnpackFn[0xb04] = &GctFormatTranslateV35::blockToFibres;                     // Leaf3JetPosEtaU1: JF2 Raw Input
    m_blockUnpackFn[0xb08] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf3JetPosEtaU1: JF3 Input
    m_blockUnpackFn[0xb09] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetPosEtaU1: JF3 Shared Received
    m_blockUnpackFn[0xb0a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetPosEtaU1: JF3 Shared Sent
    m_blockUnpackFn[0xb0b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf3JetPosEtaU1: JF3 Output
    m_blockUnpackFn[0xb0c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf3JetPosEtaU1: JF3 Raw Input
    m_blockUnpackFn[0xb80] = &GctFormatTranslateV35::blockDoNothing;                    // Leaf3JetPosEtaU2: Eta0 Input
    m_blockUnpackFn[0xb84] = &GctFormatTranslateV35::blockToFibres;                     // Leaf3JetPosEtaU2: Eta0 Raw Input
    m_blockUnpackFn[0xb88] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf3JetPosEtaU2: JF1 Input
    m_blockUnpackFn[0xb89] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetPosEtaU2: JF1 Shared Received
    m_blockUnpackFn[0xb8a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetPosEtaU2: JF1 Shared Sent
    m_blockUnpackFn[0xb8b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf3JetPosEtaU2: JF1 Output
    m_blockUnpackFn[0xb8c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf3JetPosEtaU2: JF1 Raw Input
    // Jet Leaf FPGAs - Negative Eta
    m_blockUnpackFn[0xd00] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf1JetNegEtaU1: JF2 Input
    m_blockUnpackFn[0xd01] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetNegEtaU1: JF2 Shared Received
    m_blockUnpackFn[0xd02] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetNegEtaU1: JF2 Shared Sent
    m_blockUnpackFn[0xd03] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf1JetNegEtaU1: JF2 Output
    m_blockUnpackFn[0xd04] = &GctFormatTranslateV35::blockToFibres;                     // Leaf1JetNegEtaU1: JF2 Raw Input
    m_blockUnpackFn[0xd08] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf1JetNegEtaU1: JF3 Input
    m_blockUnpackFn[0xd09] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetNegEtaU1: JF3 Shared Received
    m_blockUnpackFn[0xd0a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetNegEtaU1: JF3 Shared Sent
    m_blockUnpackFn[0xd0b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf1JetNegEtaU1: JF3 Output
    m_blockUnpackFn[0xd0c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf1JetNegEtaU1: JF3 Raw Input
    m_blockUnpackFn[0xd80] = &GctFormatTranslateV35::blockDoNothing;                    // Leaf1JetNegEtaU2: Eta0 Input
    m_blockUnpackFn[0xd84] = &GctFormatTranslateV35::blockToFibres;                     // Leaf1JetNegEtaU2: Eta0 Raw Input
    m_blockUnpackFn[0xd88] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf1JetNegEtaU2: JF1 Input
    m_blockUnpackFn[0xd89] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetNegEtaU2: JF1 Shared Received
    m_blockUnpackFn[0xd8a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf1JetNegEtaU2: JF1 Shared Sent
    m_blockUnpackFn[0xd8b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf1JetNegEtaU2: JF1 Output
    m_blockUnpackFn[0xd8c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf1JetNegEtaU2: JF1 Raw Input
    m_blockUnpackFn[0xe00] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf2JetNegEtaU1: JF2 Input
    m_blockUnpackFn[0xe01] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetNegEtaU1: JF2 Shared Received
    m_blockUnpackFn[0xe02] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetNegEtaU1: JF2 Shared Sent
    m_blockUnpackFn[0xe03] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf2JetNegEtaU1: JF2 Output
    m_blockUnpackFn[0xe04] = &GctFormatTranslateV35::blockToFibres;                     // Leaf2JetNegEtaU1: JF2 Raw Input
    m_blockUnpackFn[0xe08] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf2JetNegEtaU1: JF3 Input
    m_blockUnpackFn[0xe09] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetNegEtaU1: JF3 Shared Received
    m_blockUnpackFn[0xe0a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetNegEtaU1: JF3 Shared Sent
    m_blockUnpackFn[0xe0b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf2JetNegEtaU1: JF3 Output
    m_blockUnpackFn[0xe0c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf2JetNegEtaU1: JF3 Raw Input
    m_blockUnpackFn[0xe80] = &GctFormatTranslateV35::blockDoNothing;                    // Leaf2JetNegEtaU2: Eta0 Input
    m_blockUnpackFn[0xe84] = &GctFormatTranslateV35::blockToFibres;                     // Leaf2JetNegEtaU2: Eta0 Raw Input
    m_blockUnpackFn[0xe88] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf2JetNegEtaU2: JF1 Input
    m_blockUnpackFn[0xe89] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetNegEtaU2: JF1 Shared Received
    m_blockUnpackFn[0xe8a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf2JetNegEtaU2: JF1 Shared Sent
    m_blockUnpackFn[0xe8b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf2JetNegEtaU2: JF1 Output
    m_blockUnpackFn[0xe8c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf2JetNegEtaU2: JF1 Raw Input
    m_blockUnpackFn[0xf00] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf3JetNegEtaU1: JF2 Input
    m_blockUnpackFn[0xf01] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetNegEtaU1: JF2 Shared Received
    m_blockUnpackFn[0xf02] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetNegEtaU1: JF2 Shared Sent
    m_blockUnpackFn[0xf03] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf3JetNegEtaU1: JF2 Output
    m_blockUnpackFn[0xf04] = &GctFormatTranslateV35::blockToFibres;                     // Leaf3JetNegEtaU1: JF2 Raw Input
    m_blockUnpackFn[0xf08] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf3JetNegEtaU1: JF3 Input
    m_blockUnpackFn[0xf09] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetNegEtaU1: JF3 Shared Received
    m_blockUnpackFn[0xf0a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetNegEtaU1: JF3 Shared Sent
    m_blockUnpackFn[0xf0b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf3JetNegEtaU1: JF3 Output
    m_blockUnpackFn[0xf0c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf3JetNegEtaU1: JF3 Raw Input
    m_blockUnpackFn[0xf80] = &GctFormatTranslateV35::blockDoNothing;                    // Leaf3JetNegEtaU2: Eta0 Input
    m_blockUnpackFn[0xf84] = &GctFormatTranslateV35::blockToFibres;                     // Leaf3JetNegEtaU2: Eta0 Raw Input
    m_blockUnpackFn[0xf88] = &GctFormatTranslateV35::blockToRctCaloRegions;             // Leaf3JetNegEtaU2: JF1 Input
    m_blockUnpackFn[0xf89] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetNegEtaU2: JF1 Shared Received
    m_blockUnpackFn[0xf8a] = &GctFormatTranslateV35::blockToGctJetPreCluster;           // Leaf3JetNegEtaU2: JF1 Shared Sent
    m_blockUnpackFn[0xf8b] = &GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster;// Leaf3JetNegEtaU2: JF1 Output
    m_blockUnpackFn[0xf8c] = &GctFormatTranslateV35::blockToFibres;                     // Leaf3JetNegEtaU2: JF1 Raw Input


    /*** Setup RCT Em Crate Map ***/
    m_rctEmCrate[0x804] = 13;
    m_rctEmCrate[0x884] = 9;
    m_rctEmCrate[0xc04] = 4;
    m_rctEmCrate[0xc84] = 0;


    /*** Setup RCT jet crate map. ***/
    m_rctJetCrate[0x900] = 9;  // PosEta Leaf 1 JF2
    m_rctJetCrate[0x908] = 10; // PosEta Leaf 1 JF3
    m_rctJetCrate[0x988] = 17; // PosEta Leaf 1 JF1 
    m_rctJetCrate[0xa00] = 12; // PosEta Leaf 2 JF2
    m_rctJetCrate[0xa08] = 13; // PosEta Leaf 2 JF3
    m_rctJetCrate[0xa88] = 11; // PosEta Leaf 2 JF1 
    m_rctJetCrate[0xb00] = 15; // PosEta Leaf 3 JF2
    m_rctJetCrate[0xb08] = 16; // PosEta Leaf 3 JF3
    m_rctJetCrate[0xb88] = 14; // PosEta Leaf 3 JF1 
    m_rctJetCrate[0xd00] = 0;  // NegEta Leaf 1 JF2
    m_rctJetCrate[0xd08] = 1;  // NegEta Leaf 1 JF3
    m_rctJetCrate[0xd88] = 8;  // NegEta Leaf 1 JF1 
    m_rctJetCrate[0xe00] = 3;  // NegEta Leaf 2 JF2
    m_rctJetCrate[0xe08] = 4;  // NegEta Leaf 2 JF3
    m_rctJetCrate[0xe88] = 2;  // NegEta Leaf 2 JF1 
    m_rctJetCrate[0xf00] = 6;  // NegEta Leaf 3 JF2
    m_rctJetCrate[0xf08] = 7;  // NegEta Leaf 3 JF3
    m_rctJetCrate[0xf88] = 5;  // NegEta Leaf 3 JF1 


    /*** Setup Block ID map for pipeline payload positions of isolated Internal EM Cands. ***/
    m_internEmIsoBounds[0x680] = IsoBoundaryPair(8,15);
    m_internEmIsoBounds[0x800] = IsoBoundaryPair(0, 9);
    m_internEmIsoBounds[0x803] = IsoBoundaryPair(0, 1);
    m_internEmIsoBounds[0x880] = IsoBoundaryPair(0, 7);
    m_internEmIsoBounds[0x883] = IsoBoundaryPair(0, 1);
    m_internEmIsoBounds[0xc00] = IsoBoundaryPair(0, 9);
    m_internEmIsoBounds[0xc03] = IsoBoundaryPair(0, 1);
    m_internEmIsoBounds[0xc80] = IsoBoundaryPair(0, 7);
    m_internEmIsoBounds[0xc83] = IsoBoundaryPair(0, 1);
  }
}

GctFormatTranslateV35::~GctFormatTranslateV35()
{
}

GctBlockHeader GctFormatTranslateV35::generateBlockHeader(const unsigned char * data) const
{
  // Turn the four 8-bit header words into the full 32-bit header.
  uint32_t hdr = data[0] + (data[1]<<8) + (data[2]<<16) + (data[3]<<24);

//  Bit mapping of V35 header:
//  --------------------------
//  11:0   => block_id  Unique pipeline identifier.
//   - 3:0    =>> pipe_id There can be up to 16 different pipelines per FPGA.
//   - 6:4    =>> reserved  Do not use yet. Set to zero.
//   - 11:7   =>> fpga geograpical add  The VME geographical address of the FPGA.
//  15:12  => event_id  Determined locally.  Not reset by Resync.
//  19:16  => number_of_time_samples  If time samples 15 or more then value = 15.
//  31:20  => event_bcid  The bunch crossing the data was recorded.

  unsigned blockId = hdr & 0xfff;
  unsigned blockLength = 0;  // Set to zero until we know it's a valid block
  unsigned nSamples = (hdr>>16) & 0xf;
  unsigned bxId = (hdr>>20) & 0xfff;
  unsigned eventId = (hdr>>12) & 0xf;
  bool valid = (blockLengthMap().find(blockId) != blockLengthMap().end());

  if(valid) { blockLength = blockLengthMap().find(blockId)->second; }
  
  return GctBlockHeader(blockId, blockLength, nSamples, bxId, eventId, valid);  
}

// conversion
bool GctFormatTranslateV35::convertBlock(const unsigned char * data, const GctBlockHeader& hdr)
{
  // if the block has no time samples, don't bother with it.
  if ( hdr.nSamples() < 1 ) { return true; }

  if(!checkBlock(hdr)) { return false; }  // Check the block to see if it's possible to unpack.

  // The header validity check above will protect against
  // the map::find() method returning the end of the map,
  // assuming the block header definitions are up-to-date.
  (this->*m_blockUnpackFn.find(hdr.blockId())->second)(data, hdr);  // Calls the correct unpack function, based on block ID.
  
  return true;
}


// PROTECTED METHODS

uint32_t GctFormatTranslateV35::generateRawHeader(const uint32_t blockId,
                                                  const uint32_t nSamples,
                                                  const uint32_t bxId,
                                                  const uint32_t eventId) const
{
  //  Bit mapping of V35 header:
  //  --------------------------
  //  11:0   => block_id  Unique pipeline identifier.
  //   - 3:0    =>> pipe_id There can be up to 16 different pipelines per FPGA.
  //   - 6:4    =>> reserved  Do not use yet. Set to zero.
  //   - 11:7   =>> fpga geograpical add  The VME geographical address of the FPGA.
  //  15:12  => event_id  Determined locally.  Not reset by Resync.
  //  19:16  => number_of_time_samples  If time samples 15 or more then value = 15.
  //  31:20  => event_bxId  The bunch crossing the data was recorded.

  return ((bxId & 0xfff) << 20) | ((nSamples & 0xf) << 16) | ((eventId & 0xf) << 12) | (blockId & 0xfff);
}


// PRIVATE METHODS

// Output EM Candidates unpacking
void GctFormatTranslateV35::blockToGctEmCandsAndEnergySums(const unsigned char * d, const GctBlockHeader& hdr)
{
  const unsigned int id = hdr.blockId();
  const unsigned int nSamples = hdr.nSamples();

  // Re-interpret pointer.  p16 will be pointing at the 16 bit word that
  // contains the rank0 non-isolated electron of the zeroth time-sample.
  const uint16_t * p16 = reinterpret_cast<const uint16_t *>(d);

  // UNPACK EM CANDS

  const unsigned int emCandCategoryOffset = nSamples * 4;  // Offset to jump from the non-iso electrons to the isolated ones.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  unsigned int samplesToUnpack = 1;
  if(!hltMode()) { samplesToUnpack = nSamples; }  // Only if not running in HLT mode do we want more than 1 timesample. 

  for (unsigned int iso=0; iso<2; ++iso)  // loop over non-iso/iso candidate pairs
  {
    bool isoFlag = (iso==1);

    // Get the correct collection to put them in.
    L1GctEmCandCollection* em;
    if (isoFlag) { em = colls()->gctIsoEm(); }
    else { em = colls()->gctNonIsoEm(); }

    for (unsigned int bx=0; bx<samplesToUnpack; ++bx) // loop over time samples
    {
      // cand0Offset will give the offset on p16 to get the rank 0 candidate
      // of the correct category and timesample.
      const unsigned int cand0Offset = iso*emCandCategoryOffset + bx*2;

      em->push_back(L1GctEmCand(p16[cand0Offset], isoFlag, id, 0, bx));  // rank0 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + timeSampleOffset], isoFlag, id, 1, bx));  // rank1 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + 1], isoFlag, id, 2, bx));  // rank2 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + timeSampleOffset + 1], isoFlag, id, 3, bx));  // rank3 electron
    }
  }

  p16 += emCandCategoryOffset * 2;  // Move the pointer over the data we've already unpacked.

  // UNPACK ENERGY SUMS
  // NOTE: we are only unpacking one timesample of these currently!

  colls()->gctEtTot()->push_back(L1GctEtTotal(p16[0]));  // Et total (timesample 0).
  colls()->gctEtHad()->push_back(L1GctEtHad(p16[1]));  // Et hadronic (timesample 0).

  // 32-bit pointer for getting Missing Et.
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(p16);

  colls()->gctEtMiss()->push_back(L1GctEtMiss(p32[nSamples])); // Et Miss (timesample 0).
}

void GctFormatTranslateV35::blockToGctJetCandsAndCounts(const unsigned char * d, const GctBlockHeader& hdr)
{
  const unsigned int id = hdr.blockId();  // Capture block ID.
  const unsigned int nSamples = hdr.nSamples();  // Number of time-samples.

  // Re-interpret block payload pointer to 16 bits so it sees one candidate at a time.
  // p16 points to the start of the block payload, at the rank0 tau jet candidate.
  const uint16_t * p16 = reinterpret_cast<const uint16_t *>(d);

  // UNPACK JET CANDS

  const unsigned int jetCandCategoryOffset = nSamples * 4;  // Offset to jump from one jet category to the next.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  unsigned int samplesToUnpack = 1;
  if(!hltMode()) { samplesToUnpack = nSamples; }  // Only if not running in HLT mode do we want more than 1 timesample. 

  // Loop over the different catagories of jets
  for(unsigned int iCat = 0 ; iCat < NUM_JET_CATEGORIES ; ++iCat)
  {
    L1GctJetCandCollection * const jets = gctJets(iCat);
    assert(jets->empty()); // The supplied vector should be empty.

    bool tauflag = (iCat == TAU_JETS);
    bool forwardFlag = (iCat == FORWARD_JETS);

    // Loop over the different timesamples (bunch crossings).
    for(unsigned int bx = 0 ; bx < samplesToUnpack ; ++bx)
    {
      // cand0Offset will give the offset on p16 to get the rank 0 Jet Cand of the correct category and timesample.
      const unsigned int cand0Offset = iCat*jetCandCategoryOffset + bx*2;

      // Rank 0 Jet.
      jets->push_back(L1GctJetCand(p16[cand0Offset], tauflag, forwardFlag, id, 0, bx));
      // Rank 1 Jet.
      jets->push_back(L1GctJetCand(p16[cand0Offset + timeSampleOffset], tauflag, forwardFlag, id, 1, bx));
      // Rank 2 Jet.
      jets->push_back(L1GctJetCand(p16[cand0Offset + 1],  tauflag, forwardFlag, id, 2, bx));
      // Rank 3 Jet.
      jets->push_back(L1GctJetCand(p16[cand0Offset + timeSampleOffset + 1], tauflag, forwardFlag, id, 3, bx));
    }
  }

  p16 += NUM_JET_CATEGORIES * jetCandCategoryOffset; // Move the pointer over the data we've already unpacked.

  // NOW UNPACK: HFBitCounts, HFRingEtSums (no Missing Ht yet)
  // NOTE: we are only unpacking one timesample of these currently!

  // Re-interpret block payload pointer to 32 bits so it sees six jet counts at a time.
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(p16);

  // Channel 0 carries both HF counts and sums
  colls()->gctHfBitCounts()->push_back(L1GctHFBitCounts::fromConcHFBitCounts(id,6,0,p32[0])); 
  colls()->gctHfRingEtSums()->push_back(L1GctHFRingEtSums::fromConcRingSums(id,6,0,p32[0]));
  // Skip channel 1 for now. Later this may carry MHT would be accessed as p32[nSamples]
}

// Internal EM Candidates unpacking
void GctFormatTranslateV35::blockToGctInternEmCand(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal EM Cands"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int numCandPairs = hdr.blockLength();

  // Debug assertion to prevent problems if definitions not up to date.
  assert(internEmIsoBounds().find(id) != internEmIsoBounds().end());  

  unsigned int lowerIsoPairBound = internEmIsoBounds()[id].first;
  unsigned int upperIsoPairBound = internEmIsoBounds()[id].second;

  // Re-interpret pointer to 16 bits so it sees one candidate at a time.
  const uint16_t * p = reinterpret_cast<const uint16_t *>(d);

  // Loop over timesamples (i.e. bunch crossings)
  for(unsigned int bx=0; bx < nSamples; ++bx)
  {
    // Loop over candidate pairs (i.e. each iteration unpacks a pair of candidates)
    for(unsigned int candPair = 0 ; candPair < numCandPairs ; ++candPair)
    {
      // Is the candidate electron pair an isolated pair or not?
      bool iso = ((candPair>=lowerIsoPairBound) && (candPair<=upperIsoPairBound));
      
      // Loop over the two electron candidates in each pair
      for(unsigned int i = 0 ; i < 2 ; ++i)
      { 
        unsigned offset = 2*(bx + candPair*nSamples) + i;
        uint16_t candRaw = p[offset]; 
        colls()->gctInternEm()->push_back( L1GctInternEmCand(candRaw, iso, id, candPair*2 + i, bx) );
      }
    }
  }
}


// Input EM Candidates unpacking
// this is the last time I deal the RCT bit assignment travesty!!!
void GctFormatTranslateV35::blockToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of RCT EM Cands"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // re-interpret pointer
  const uint16_t * p = reinterpret_cast<const uint16_t *>(d);

  // arrays of source card data
  uint16_t sfp[2][4]; // [ cycle ] [ SFP ]
  uint16_t eIsoRank[4];
  uint16_t eIsoCard[4];
  uint16_t eIsoRgn[4];
  uint16_t eNonIsoRank[4];
  uint16_t eNonIsoCard[4];
  uint16_t eNonIsoRgn[4];
  uint16_t MIPbits[7][2];
  uint16_t QBits[7][2];

  unsigned int bx = 0;

  // loop over crates
  for (unsigned int crate=rctEmCrateMap()[id]; crate<rctEmCrateMap()[id]+length/3; ++crate) {

    // read SC SFP words
    for (unsigned short iSfp=0 ; iSfp<4 ; ++iSfp) {
      for (unsigned short cyc=0 ; cyc<2 ; ++cyc) {
        if (iSfp==0) { sfp[cyc][iSfp] = 0; } // muon bits
        else {                               // EM candidate
          sfp[cyc][iSfp] = *p;
          ++p;
        }
      }
      p = p + 2*(nSamples-1);
    }

    // fill SC arrays
    srcCardRouting().SFPtoEMU(eIsoRank, eIsoCard, eIsoRgn, eNonIsoRank, eNonIsoCard, eNonIsoRgn, MIPbits, QBits, sfp);
    
    // create EM cands
    for (unsigned short int i=0; i<4; ++i) {
      colls()->rctEm()->push_back( L1CaloEmCand( eIsoRank[i], eIsoRgn[i], eIsoCard[i], crate, true, i, bx) );
    }
    for (unsigned short int i=0; i<4; ++i) {
      colls()->rctEm()->push_back( L1CaloEmCand( eNonIsoRank[i], eNonIsoRgn[i], eNonIsoCard[i], crate, false, i, bx) );
    }
  }
}

// Input RCT region unpacking
void GctFormatTranslateV35::blockToRctCaloRegions(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of RCT Regions"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Debug assertion to prevent problems if definitions not up to date.
  assert(rctJetCrateMap().find(id) != rctJetCrateMap().end());  
  
  // get crate (need this to get ieta and iphi)
  unsigned int crate=rctJetCrateMap()[id];

  // re-interpret pointer
  const uint16_t * p = reinterpret_cast<const uint16_t *>(d);
  
  // eta and phi
  unsigned int ieta; 
  unsigned int iphi; 
  
  for (unsigned int i=0; i<length; ++i) { 
    for (uint16_t bx=0; bx<nSamples; ++bx) {
      if (i>0) {
        if (crate<9){ // negative eta
          ieta = 11-i; 
          iphi = 2*((11-crate)%9);
        } else {      // positive eta
          ieta = 10+i;
          iphi = 2*((20-crate)%9);
        }        
        // First region is phi=0
        colls()->rctCalo()->push_back( L1CaloRegion::makeRegionFromUnpacker(*p, ieta, iphi, id, i, bx) );
        ++p;
        // Second region is phi=1
        if (iphi>0){
          iphi-=1;
        } else {
          iphi = 17;
        }
        colls()->rctCalo()->push_back( L1CaloRegion::makeRegionFromUnpacker(*p, ieta, iphi, id, i, bx) );
        ++p;
      } else { // Skip the first two regions which are duplicates. 
        ++p;
        ++p;
      }
    }
  } 
}  

// Fibre unpacking
void GctFormatTranslateV35::blockToFibres(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of GCT Fibres"; return; }
  
  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // re-interpret pointer
  const uint32_t * p = reinterpret_cast<const uint32_t *>(d);

  for (unsigned int i=0; i<length; ++i) {
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      colls()->gctFibres()->push_back( L1GctFibreWord(*p, id, i, bx) );
      ++p;
    }
  } 
}

void GctFormatTranslateV35::blockToFibresAndToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr)
{
  this->blockToRctEmCand(d, hdr);
  this->blockToFibres(d, hdr);
}

void GctFormatTranslateV35::blockToGctInternEtSums(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!                                                                                                                           
  
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Et Sums"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Re-interpret pointer to 32 bits 
  const uint32_t * p = reinterpret_cast<const uint32_t *>(d);

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)                                                                                                                            
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      colls()->gctInternEtSums()->push_back(L1GctInternEtSum::fromTotalEtOrHt(id,i,bx,*p));
      ++p;
    }
  }
}

void GctFormatTranslateV35::blockToGctInternEtSumsAndJetCluster(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Jet Cands"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Re-interpret pointer to 32 bits 
  const uint32_t * p = reinterpret_cast<const uint32_t *>(d);

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      if (i<2) colls()->gctInternEtSums()->push_back(L1GctInternEtSum::fromJetMissEt(id,i,bx,*p));
      if (i==3){
        colls()->gctInternEtSums()->push_back(L1GctInternEtSum::fromJetTotEt(id,i,bx,*p));
        colls()->gctInternEtSums()->push_back(L1GctInternEtSum::fromJetTotHt(id,i,bx,*p));
      } 
      if (i>4) colls()->gctInternJets()->push_back(L1GctInternJetData::fromJetCluster(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
    }  
  }
}

void GctFormatTranslateV35::blockToGctTrigObjects(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Jet Cands"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Re-interpret pointer to 16 bits so it sees one candidate at a time.
  const uint16_t * p = reinterpret_cast<const uint16_t *>(d);

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      colls()->gctInternJets()->push_back( L1GctInternJetData::fromGctTrigObject(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
      colls()->gctInternJets()->push_back( L1GctInternJetData::fromGctTrigObject(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
    }
  } 
}

void GctFormatTranslateV35::blockToGctJetClusterMinimal(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Jet Cands"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Re-interpret pointer to 16 bits so it sees one candidate at a time.
  const uint16_t * p = reinterpret_cast<const uint16_t *>(d);

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      colls()->gctInternJets()->push_back( L1GctInternJetData::fromJetClusterMinimal(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
      colls()->gctInternJets()->push_back( L1GctInternJetData::fromJetClusterMinimal(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
    }
  } 
}

void GctFormatTranslateV35::blockToGctJetPreCluster(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Jet Cands"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Re-interpret pointer to 16 bits so it sees one candidate at a time.
  const uint16_t * p = reinterpret_cast<const uint16_t *>(d);

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      colls()->gctInternJets()->push_back( L1GctInternJetData::fromJetPreCluster(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
      colls()->gctInternJets()->push_back( L1GctInternJetData::fromJetPreCluster(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
    }
  } 
}

void GctFormatTranslateV35::blockToGctInternRingSums(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal HF ring data"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Re-interpret pointer to 32 bits 
  const uint32_t * p = reinterpret_cast<const uint32_t *>(d);

  for (unsigned int i=0; i<length/2; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      colls()->gctInternHFData()->push_back(L1GctInternHFData::fromConcRingSums(id,i,bx,*p));
      ++p;
    }
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      colls()->gctInternHFData()->push_back(L1GctInternHFData::fromConcBitCounts(id,i,bx,*p));
      ++p;
    }  
  }
}

void GctFormatTranslateV35::blockToGctWheelInputInternEtAndRingSums(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of wheel input internal Et sums and HF ring data"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Re-interpret pointer to 32 bits 
  const uint32_t * p = reinterpret_cast<const uint32_t *>(d);

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      if (i<3){
        colls()->gctInternEtSums()->push_back(L1GctInternEtSum::fromTotalEtOrHt(id,i,bx,*p));
      } else if (i>2 && i<9) {
        colls()->gctInternEtSums()->push_back(L1GctInternEtSum::fromMissEtxOrEty(id,i,bx,*p));
      } else if (i>8 && i<15) {
        colls()->gctInternHFData()->push_back(L1GctInternHFData::fromWheelRingSums(id,i,bx,*p));
      } else if (i>14){
        colls()->gctInternHFData()->push_back(L1GctInternHFData::fromWheelBitCounts(id,i,bx,*p));
      }
      ++p;
    }
  }
}

void GctFormatTranslateV35::blockToGctWheelOutputInternEtAndRingSums(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of wheel output internal Et sums and HF ring data"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // Re-interpret pointer to 32 bits 
  const uint32_t * p = reinterpret_cast<const uint32_t *>(d);

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      if (i<1){
        colls()->gctInternEtSums()->push_back(L1GctInternEtSum::fromTotalEtOrHt(id,i,bx,*p));
      } else if (i>0 && i<3) {
        colls()->gctInternEtSums()->push_back(L1GctInternEtSum::fromMissEtxOrEty(id,i,bx,*p));
      } else if (i>2 && i<5) {
        colls()->gctInternHFData()->push_back(L1GctInternHFData::fromWheelRingSums(id,i,bx,*p));
      } else if (i>4){
        colls()->gctInternHFData()->push_back(L1GctInternHFData::fromWheelBitCounts(id,i,bx,*p));
      }
      ++p;
    }
  }
}
