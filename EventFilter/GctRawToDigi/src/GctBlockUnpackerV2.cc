#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerV2.h"

// C++ headers
#include <iostream>
#include <cassert>

// Framework headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Namespace resolution
using std::cout;
using std::endl;
using std::pair;

// INITIALISE STATIC VARIABLES
GctBlockUnpackerV2::RctCrateMap GctBlockUnpackerV2::rctCrate_ = GctBlockUnpackerV2::RctCrateMap();
GctBlockUnpackerV2::RctJetCrateMap GctBlockUnpackerV2::rctJetCrate_ = GctBlockUnpackerV2::RctJetCrateMap();
GctBlockUnpackerV2::BlockIdToEmCandIsoBoundMap GctBlockUnpackerV2::internEmIsoBounds_ = GctBlockUnpackerV2::BlockIdToEmCandIsoBoundMap();
GctBlockUnpackerV2::BlockIdToUnpackFnMap GctBlockUnpackerV2::blockUnpackFn_ = GctBlockUnpackerV2::BlockIdToUnpackFnMap();

GctBlockUnpackerV2::GctBlockUnpackerV2(bool hltMode):
  GctBlockUnpackerBase(hltMode)
{
  static bool initClass = true;

  if(initClass)
  {
    initClass = false;

    // Setup RCT crate map.
    rctCrate_[0x804] = 13;
    rctCrate_[0x884] = 9;
    rctCrate_[0xc04] = 4;
    rctCrate_[0xc84] = 0;

    // Setup RCT jet crate map.
    rctJetCrate_[0x900] = 9;  // Leaf 1 JF2
    rctJetCrate_[0x908] = 10; // Leaf 1 JF3
    rctJetCrate_[0x988] = 17; // Leaf 1 JF1 

    rctJetCrate_[0xa00] = 12; // Leaf 2 JF2
    rctJetCrate_[0xa08] = 13; // Leaf 2 JF3
    rctJetCrate_[0xa88] = 11; // Leaf 2 JF1 

    rctJetCrate_[0xb00] = 15; // Leaf 3 JF2
    rctJetCrate_[0xb08] = 16; // Leaf 3 JF3
    rctJetCrate_[0xb88] = 14; // Leaf 3 JF1 

    rctJetCrate_[0xd00] = 0; // Leaf 1 JF2
    rctJetCrate_[0xd08] = 1; // Leaf 1 JF3
    rctJetCrate_[0xd88] = 8; // Leaf 1 JF1 

    rctJetCrate_[0xe00] = 3; // Leaf 2 JF2
    rctJetCrate_[0xe08] = 4; // Leaf 2 JF3
    rctJetCrate_[0xe88] = 2; // Leaf 2 JF1 

    rctJetCrate_[0xf00] = 6; // Leaf 3 JF2
    rctJetCrate_[0xf08] = 7; // Leaf 3 JF3
    rctJetCrate_[0xf88] = 5; // Leaf 3 JF1 

    // Setup Block ID map for pipeline payload positions of isolated Internal EM Cands.
    internEmIsoBounds_[0x680] = IsoBoundaryPair(8,15);
    internEmIsoBounds_[0x800] = IsoBoundaryPair(0, 9);
    internEmIsoBounds_[0x803] = IsoBoundaryPair(0, 1);
    internEmIsoBounds_[0x880] = IsoBoundaryPair(0, 7);
    internEmIsoBounds_[0x883] = IsoBoundaryPair(0, 1);
    internEmIsoBounds_[0xc00] = IsoBoundaryPair(0, 9);
    internEmIsoBounds_[0xc03] = IsoBoundaryPair(0, 1);
    internEmIsoBounds_[0xc80] = IsoBoundaryPair(0, 7);
    internEmIsoBounds_[0xc83] = IsoBoundaryPair(0, 1);

    // SETUP BLOCK UNPACK FUNCTION MAP
    //  -- Miscellaneous Blocks
    blockUnpackFn_[0x000] = &GctBlockUnpackerV2::blockDoNothing;                    // NULL
    blockUnpackFn_[0x0ff] = &GctBlockUnpackerV2::blockToAllRctCaloRegions;          // Temporary hack: All RCT Calo Regions for CMSSW pack/unpack
    //  -- ConcJet FPGA                                                             
    blockUnpackFn_[0x580] = &GctBlockUnpackerV2::blockToGctTrigObjects;             // ConcJet: Input TrigPathA (Jet Cands)
    blockUnpackFn_[0x581] = &GctBlockUnpackerV2::blockToGctInternRingSums;          // ConcJet: Input TrigPathB (HF Rings)
    blockUnpackFn_[0x583] = &GctBlockUnpackerV2::blockToGctJetCandsAndCounts;       // ConcJet: Jet Cands and Counts Output to GT
    blockUnpackFn_[0x587] = &GctBlockUnpackerV2::blockDoNothing;                    // ConcJet: BX & Orbit Info
    //  -- ConcElec FPGA                                                            
    blockUnpackFn_[0x680] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // ConcElec: Input TrigPathA (EM Cands)
    blockUnpackFn_[0x681] = &GctBlockUnpackerV2::blockToGctInternEtSums;            // ConcElec: Input TrigPathB (Et Sums)
    blockUnpackFn_[0x682] = &GctBlockUnpackerV2::blockToGctInternEtSums;            // ConcElec: Input TrigPathC (Ht Sums)
    blockUnpackFn_[0x683] = &GctBlockUnpackerV2::blockToGctEmCandsAndEnergySums;    // ConcElec: EM Cands and Energy Sums Output to GT
    blockUnpackFn_[0x686] = &GctBlockUnpackerV2::blockDoNothing;                    // ConcElec: Test (GT Serdes Loopback)
    blockUnpackFn_[0x687] = &GctBlockUnpackerV2::blockDoNothing;                    // ConcElec: BX & Orbit Info
    //  -- Electron Leaf FPGAs                                                      
    blockUnpackFn_[0x800] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // Leaf0ElecPosEtaU1: Sort Input
    blockUnpackFn_[0x803] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // Leaf0ElecPosEtaU1: Sort Output
    blockUnpackFn_[0x804] = &GctBlockUnpackerV2::blockToFibresAndToRctEmCand;       // Leaf0ElecPosEtaU1: Raw Input
    blockUnpackFn_[0x880] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // Leaf0ElecPosEtaU2: Sort Input
    blockUnpackFn_[0x883] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // Leaf0ElecPosEtaU2: Sort Output
    blockUnpackFn_[0x884] = &GctBlockUnpackerV2::blockToFibresAndToRctEmCand;       // Leaf0ElecPosEtaU2: Raw Input
    blockUnpackFn_[0xc00] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // Leaf0ElecNegEtaU1: Sort Input
    blockUnpackFn_[0xc03] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // Leaf0ElecNegEtaU1: Sort Output
    blockUnpackFn_[0xc04] = &GctBlockUnpackerV2::blockToFibresAndToRctEmCand;       // Leaf0ElecNegEtaU1: Raw Input
    blockUnpackFn_[0xc80] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // Leaf0ElecNegEtaU2: Sort Input
    blockUnpackFn_[0xc83] = &GctBlockUnpackerV2::blockToGctInternEmCand;            // Leaf0ElecNegEtaU2: Sort Output
    blockUnpackFn_[0xc84] = &GctBlockUnpackerV2::blockToFibresAndToRctEmCand;       // Leaf0ElecNegEtaU2: Raw Input
    //  -- Wheel Pos-eta Jet FPGA                                                   
    blockUnpackFn_[0x300] = &GctBlockUnpackerV2::blockToGctJetClusterMinimal;       // WheelPosEtaJet: Input TrigPathA
    blockUnpackFn_[0x303] = &GctBlockUnpackerV2::blockToGctTrigObjects;             // WheelPosEtaJet: Output TrigPathA
    blockUnpackFn_[0x306] = &GctBlockUnpackerV2::blockDoNothing;                    // WheelPosEtaJet: Test (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    blockUnpackFn_[0x307] = &GctBlockUnpackerV2::blockDoNothing;                    // WheelPosEtaJet: Info (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    //  -- Wheel Pos-eta Energy FPGA                                                
    blockUnpackFn_[0x380] = &GctBlockUnpackerV2::blockToGctWheelInputInternEtAndRingSums;     // WheelPosEtaEnergy: Input TrigPathA (Et)
    blockUnpackFn_[0x381] = &GctBlockUnpackerV2::blockToGctInternEtSums;            // WheelPosEtaEnergy: Input TrigPathB (Ht)
    blockUnpackFn_[0x383] = &GctBlockUnpackerV2::blockToGctWheelOutputInternEtAndRingSums;     // WheelPosEtaEnergy: Output TrigPathA (Et)
    blockUnpackFn_[0x385] = &GctBlockUnpackerV2::blockToGctInternEtSums;            // WheelPosEtaEnergy: Output TrigPathB (Ht)
    blockUnpackFn_[0x386] = &GctBlockUnpackerV2::blockDoNothing;                    // WheelPosEtaEnergy: Test
    blockUnpackFn_[0x387] = &GctBlockUnpackerV2::blockDoNothing;                    // WheelPosEtaEnergy: BX & Orbit Info   (Potential data incompatibility between V24/V25 where block length=4, and V27.1 where block length=6)
    //  -- Wheel Neg-eta Jet FPGA                                                   
    blockUnpackFn_[0x700] = &GctBlockUnpackerV2::blockToGctJetClusterMinimal;       // WheelNegEtaJet: Input TrigPathA
    blockUnpackFn_[0x703] = &GctBlockUnpackerV2::blockToGctTrigObjects;             // WheelNegEtaJet: Output TrigPathA
    blockUnpackFn_[0x706] = &GctBlockUnpackerV2::blockDoNothing;                    // WheelNegEtaJet: Test (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    blockUnpackFn_[0x707] = &GctBlockUnpackerV2::blockDoNothing;                    // WheelNegEtaJet: Info (deprecated)  (Doesn't exist in V27.1 format, but does in V24 & V25, so keep for CRUZET2 data compatibility reasons)
    //  -- Wheel Neg-eta Energy FPGA                                                
    blockUnpackFn_[0x780] = &GctBlockUnpackerV2::blockToGctWheelInputInternEtAndRingSums;     // WheelNegEtaEnergy: Input TrigPathA (Et)
    blockUnpackFn_[0x781] = &GctBlockUnpackerV2::blockToGctInternEtSums;            // WheelNegEtaEnergy: Input TrigPathB (Ht)
    blockUnpackFn_[0x783] = &GctBlockUnpackerV2::blockToGctWheelOutputInternEtAndRingSums;     // WheelNegEtaEnergy: Output TrigPathA (Et)
    blockUnpackFn_[0x785] = &GctBlockUnpackerV2::blockToGctInternEtSums;            // WheelNegEtaEnergy: Output TrigPathB (Ht)
    blockUnpackFn_[0x786] = &GctBlockUnpackerV2::blockDoNothing;                    // WheelNegEtaEnergy: Test
    blockUnpackFn_[0x787] = &GctBlockUnpackerV2::blockDoNothing;                    // WheelNegEtaEnergy: BX & Orbit Info   (Potential data incompatibility between V24/V25 where block length=4, and V27.1 where block length=6)
    //  -- Jet Leaf FPGAs - Positive Eta                                            
    blockUnpackFn_[0x900] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf1JetPosEtaU1: JF2 Input
    blockUnpackFn_[0x901] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetPosEtaU1: JF2 Shared Recieved
    blockUnpackFn_[0x902] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetPosEtaU1: JF2 Shared Sent
    blockUnpackFn_[0x903] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf1JetPosEtaU1: JF2 Output
    blockUnpackFn_[0x904] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf1JetPosEtaU1: JF2 Raw Input
    blockUnpackFn_[0x908] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf1JetPosEtaU1: JF3 Input
    blockUnpackFn_[0x909] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetPosEtaU1: JF3 Shared Recieved
    blockUnpackFn_[0x90a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetPosEtaU1: JF3 Shared Sent
    blockUnpackFn_[0x90b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf1JetPosEtaU1: JF3 Output
    blockUnpackFn_[0x90c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf1JetPosEtaU1: JF3 Raw Input
    blockUnpackFn_[0x980] = &GctBlockUnpackerV2::blockDoNothing;                    // Leaf1JetPosEtaU2: Eta0 Input
    blockUnpackFn_[0x984] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf1JetPosEtaU2: Eta0 Raw Input
    blockUnpackFn_[0x988] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf1JetPosEtaU2: JF1 Input
    blockUnpackFn_[0x989] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetPosEtaU2: JF1 Shared Recieved
    blockUnpackFn_[0x98a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetPosEtaU2: JF1 Shared Sent
    blockUnpackFn_[0x98b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf1JetPosEtaU2: JF1 Output
    blockUnpackFn_[0x98c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf1JetPosEtaU2: JF1 Raw Input
    blockUnpackFn_[0xa00] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf2JetPosEtaU1: JF2 Input
    blockUnpackFn_[0xa01] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetPosEtaU1: JF2 Shared Recieved
    blockUnpackFn_[0xa02] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetPosEtaU1: JF2 Shared Sent
    blockUnpackFn_[0xa03] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf2JetPosEtaU1: JF2 Output
    blockUnpackFn_[0xa04] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf2JetPosEtaU1: JF2 Raw Input
    blockUnpackFn_[0xa08] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf2JetPosEtaU1: JF3 Input
    blockUnpackFn_[0xa09] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetPosEtaU1: JF3 Shared Recieved
    blockUnpackFn_[0xa0a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetPosEtaU1: JF3 Shared Sent
    blockUnpackFn_[0xa0b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf2JetPosEtaU1: JF3 Output
    blockUnpackFn_[0xa0c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf2JetPosEtaU1: JF3 Raw Input
    blockUnpackFn_[0xa80] = &GctBlockUnpackerV2::blockDoNothing;                    // Leaf2JetPosEtaU2: Eta0 Input
    blockUnpackFn_[0xa84] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf2JetPosEtaU2: Eta0 Raw Input
    blockUnpackFn_[0xa88] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf2JetPosEtaU2: JF1 Input
    blockUnpackFn_[0xa89] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetPosEtaU2: JF1 Shared Recieved
    blockUnpackFn_[0xa8a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetPosEtaU2: JF1 Shared Sent
    blockUnpackFn_[0xa8b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf2JetPosEtaU2: JF1 Output
    blockUnpackFn_[0xa8c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf2JetPosEtaU2: JF1 Raw Input
    blockUnpackFn_[0xb00] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf3JetPosEtaU1: JF2 Input
    blockUnpackFn_[0xb01] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetPosEtaU1: JF2 Shared Recieved
    blockUnpackFn_[0xb02] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetPosEtaU1: JF2 Shared Sent
    blockUnpackFn_[0xb03] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf3JetPosEtaU1: JF2 Output
    blockUnpackFn_[0xb04] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf3JetPosEtaU1: JF2 Raw Input
    blockUnpackFn_[0xb08] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf3JetPosEtaU1: JF3 Input
    blockUnpackFn_[0xb09] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetPosEtaU1: JF3 Shared Recieved
    blockUnpackFn_[0xb0a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetPosEtaU1: JF3 Shared Sent
    blockUnpackFn_[0xb0b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf3JetPosEtaU1: JF3 Output
    blockUnpackFn_[0xb0c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf3JetPosEtaU1: JF3 Raw Input
    blockUnpackFn_[0xb80] = &GctBlockUnpackerV2::blockDoNothing;                    // Leaf3JetPosEtaU2: Eta0 Input
    blockUnpackFn_[0xb84] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf3JetPosEtaU2: Eta0 Raw Input
    blockUnpackFn_[0xb88] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf3JetPosEtaU2: JF1 Input
    blockUnpackFn_[0xb89] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetPosEtaU2: JF1 Shared Recieved
    blockUnpackFn_[0xb8a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetPosEtaU2: JF1 Shared Sent
    blockUnpackFn_[0xb8b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf3JetPosEtaU2: JF1 Output
    blockUnpackFn_[0xb8c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf3JetPosEtaU2: JF1 Raw Input
    //  -- Jet Leaf FPGAs - Negative Eta                                            
    blockUnpackFn_[0xd00] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf1JetNegEtaU1: JF2 Input
    blockUnpackFn_[0xd01] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetNegEtaU1: JF2 Shared Recieved
    blockUnpackFn_[0xd02] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetNegEtaU1: JF2 Shared Sent
    blockUnpackFn_[0xd03] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf1JetNegEtaU1: JF2 Output
    blockUnpackFn_[0xd04] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf1JetNegEtaU1: JF2 Raw Input
    blockUnpackFn_[0xd08] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf1JetNegEtaU1: JF3 Input
    blockUnpackFn_[0xd09] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetNegEtaU1: JF3 Shared Recieved
    blockUnpackFn_[0xd0a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetNegEtaU1: JF3 Shared Sent
    blockUnpackFn_[0xd0b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf1JetNegEtaU1: JF3 Output
    blockUnpackFn_[0xd0c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf1JetNegEtaU1: JF3 Raw Input
    blockUnpackFn_[0xd80] = &GctBlockUnpackerV2::blockDoNothing;                    // Leaf1JetNegEtaU2: Eta0 Input
    blockUnpackFn_[0xd84] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf1JetNegEtaU2: Eta0 Raw Input
    blockUnpackFn_[0xd88] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf1JetNegEtaU2: JF1 Input
    blockUnpackFn_[0xd89] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetNegEtaU2: JF1 Shared Recieved
    blockUnpackFn_[0xd8a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf1JetNegEtaU2: JF1 Shared Sent
    blockUnpackFn_[0xd8b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf1JetNegEtaU2: JF1 Output
    blockUnpackFn_[0xd8c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf1JetNegEtaU2: JF1 Raw Input
    blockUnpackFn_[0xe00] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf2JetNegEtaU1: JF2 Input
    blockUnpackFn_[0xe01] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetNegEtaU1: JF2 Shared Recieved
    blockUnpackFn_[0xe02] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetNegEtaU1: JF2 Shared Sent
    blockUnpackFn_[0xe03] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf2JetNegEtaU1: JF2 Output
    blockUnpackFn_[0xe04] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf2JetNegEtaU1: JF2 Raw Input
    blockUnpackFn_[0xe08] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf2JetNegEtaU1: JF3 Input
    blockUnpackFn_[0xe09] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetNegEtaU1: JF3 Shared Recieved
    blockUnpackFn_[0xe0a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetNegEtaU1: JF3 Shared Sent
    blockUnpackFn_[0xe0b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf2JetNegEtaU1: JF3 Output
    blockUnpackFn_[0xe0c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf2JetNegEtaU1: JF3 Raw Input
    blockUnpackFn_[0xe80] = &GctBlockUnpackerV2::blockDoNothing;                    // Leaf2JetNegEtaU2: Eta0 Input
    blockUnpackFn_[0xe84] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf2JetNegEtaU2: Eta0 Raw Input
    blockUnpackFn_[0xe88] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf2JetNegEtaU2: JF1 Input
    blockUnpackFn_[0xe89] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetNegEtaU2: JF1 Shared Recieved
    blockUnpackFn_[0xe8a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf2JetNegEtaU2: JF1 Shared Sent
    blockUnpackFn_[0xe8b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf2JetNegEtaU2: JF1 Output
    blockUnpackFn_[0xe8c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf2JetNegEtaU2: JF1 Raw Input
    blockUnpackFn_[0xf00] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf3JetNegEtaU1: JF2 Input
    blockUnpackFn_[0xf01] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetNegEtaU1: JF2 Shared Recieved
    blockUnpackFn_[0xf02] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetNegEtaU1: JF2 Shared Sent
    blockUnpackFn_[0xf03] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf3JetNegEtaU1: JF2 Output
    blockUnpackFn_[0xf04] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf3JetNegEtaU1: JF2 Raw Input
    blockUnpackFn_[0xf08] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf3JetNegEtaU1: JF3 Input
    blockUnpackFn_[0xf09] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetNegEtaU1: JF3 Shared Recieved
    blockUnpackFn_[0xf0a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetNegEtaU1: JF3 Shared Sent
    blockUnpackFn_[0xf0b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf3JetNegEtaU1: JF3 Output
    blockUnpackFn_[0xf0c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf3JetNegEtaU1: JF3 Raw Input
    blockUnpackFn_[0xf80] = &GctBlockUnpackerV2::blockDoNothing;                    // Leaf3JetNegEtaU2: Eta0 Input
    blockUnpackFn_[0xf84] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf3JetNegEtaU2: Eta0 Raw Input
    blockUnpackFn_[0xf88] = &GctBlockUnpackerV2::blockToRctCaloRegions;             // Leaf3JetNegEtaU2: JF1 Input
    blockUnpackFn_[0xf89] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetNegEtaU2: JF1 Shared Recieved
    blockUnpackFn_[0xf8a] = &GctBlockUnpackerV2::blockToGctJetPreCluster;           // Leaf3JetNegEtaU2: JF1 Shared Sent
    blockUnpackFn_[0xf8b] = &GctBlockUnpackerV2::blockToGctInternEtSumsAndJetCluster;// Leaf3JetNegEtaU2: JF1 Output
    blockUnpackFn_[0xf8c] = &GctBlockUnpackerV2::blockToFibres;                     // Leaf3JetNegEtaU2: JF1 Raw Input
  }
}

GctBlockUnpackerV2::~GctBlockUnpackerV2() { }

// conversion
bool GctBlockUnpackerV2::convertBlock(const unsigned char * data, const GctBlockHeaderBase& hdr)
{
  // if the block has no time samples, don't bother with it.
  if ( hdr.nSamples() < 1 ) { return true; }

  if(!checkBlock(hdr)) { return false; }  // Check the block to see if it's possible to unpack.

  // The header validity check above will protect against
  // the map::find() method returning the end of the map,
  // assuming the block header definitions are up-to-date.
  (this->*blockUnpackFn_.find(hdr.id())->second)(data, hdr);  // Calls the correct unpack function, based on block ID.
  
  return true;
}

// Output EM Candidates unpacking
void GctBlockUnpackerV2::blockToGctEmCandsAndEnergySums(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  const unsigned int id = hdr.id();
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
    if (isoFlag) { em = gctIsoEm_; }
    else { em = gctNonIsoEm_; }

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

  gctEtTotal_->push_back(L1GctEtTotal(p16[0]));  // Et total (timesample 0).
  gctEtHad_->push_back(L1GctEtHad(p16[1]));  // Et hadronic (timesample 0).

  // 32-bit pointer for getting Missing Et.
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(p16);

  gctEtMiss_->push_back(L1GctEtMiss(p32[nSamples])); // Et Miss (timesample 0).
}

void GctBlockUnpackerV2::blockToGctJetCandsAndCounts(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  const unsigned int id = hdr.id();  // Capture block ID.
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
    assert(gctJets_.at(iCat)->empty()); // The supplied vector should be empty.

    bool tauflag = (iCat == TAU_JETS);
    bool forwardFlag = (iCat == FORWARD_JETS);

    // Loop over the different timesamples (bunch crossings).
    for(unsigned int bx = 0 ; bx < samplesToUnpack ; ++bx)
    {
      // cand0Offset will give the offset on p16 to get the rank 0 Jet Cand of the correct category and timesample.
      const unsigned int cand0Offset = iCat*jetCandCategoryOffset + bx*2;

      // Rank 0 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p16[cand0Offset], tauflag, forwardFlag, id, 0, bx));
      // Rank 1 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p16[cand0Offset + timeSampleOffset], tauflag, forwardFlag, id, 1, bx));
      // Rank 2 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p16[cand0Offset + 1],  tauflag, forwardFlag, id, 2, bx));
      // Rank 3 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p16[cand0Offset + timeSampleOffset + 1], tauflag, forwardFlag, id, 3, bx));
    }
  }

  p16 += NUM_JET_CATEGORIES * jetCandCategoryOffset; // Move the pointer over the data we've already unpacked.

  // UNPACK JET COUNTS
  // NOTE: we are only unpacking one timesample of these currently!

  // Re-interpret block payload pointer to 32 bits so it sees six jet counts at a time.
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(p16);

  gctHFBitCounts_->push_back(L1GctHFBitCounts::fromConcHFBitCounts(id,6,0,p32[0])); // Channel 0 carries both HF counts and sums
  gctHFRingEtSums_->push_back(L1GctHFRingEtSums::fromConcRingSums(id,6,0,p32[0]));  
  // Skip channel 1 for now. Later this may carry MHT would be accessed as p32[nSamples]
}
