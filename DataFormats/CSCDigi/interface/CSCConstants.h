#ifndef DataFormats_CSCDigi_CSCConstants_h
#define DataFormats_CSCDigi_CSCConstants_h

/**
 * \class CSCConstants
 *
 * Static interface to basic chamber constants.
 */

class CSCConstants {
public:
  enum DDU_Info { NUM_DDUS = 5 };

  enum CFEB_Info {
    // Run-1: Maximum number of cathode front-end boards
    MAX_CFEBS_RUN1 = 5,
    // ME1/1 cases
    NUM_CFEBS_ME1A_GANGED = 1,
    NUM_CFEBS_ME1A_UNGANGED = 3,
    NUM_CFEBS_ME1B = 4,
    NUM_CFEBS_ME11_GANGED = NUM_CFEBS_ME1A_GANGED + NUM_CFEBS_ME1B,      // 5
    NUM_CFEBS_ME11_UNGANGED = NUM_CFEBS_ME1A_UNGANGED + NUM_CFEBS_ME1B,  // 7
    // Run-2: Maximum number of cathode front-end boards
    MAX_CFEBS_RUN2 = NUM_CFEBS_ME11_UNGANGED,  // 7
    // CFEBS for the rest of the system
    NUM_CFEBS_ME12 = 5,
    NUM_CFEBS_ME13 = 4,
    NUM_CFEBS_ME21 = 5,
    NUM_CFEBS_ME22 = 5,
    NUM_CFEBS_ME31 = 5,
    NUM_CFEBS_ME32 = 5,
    NUM_CFEBS_ME41 = 5,
    NUM_CFEBS_ME42 = 5
  };

  enum FPGA_Latency { CLCT_EMUL_TIME_OFFSET = 3, ALCT_EMUL_TIME_OFFSET = 6 };

  // Numbers obtained from https://twiki.cern.ch/twiki/pub/CMS/CSCDPGGeometry/table_of_csc_properties_150730.pdf
  enum WG_Info {
    NUM_WIREGROUPS_ME11 = 48,
    NUM_WIREGROUPS_ME12 = 64,
    NUM_WIREGROUPS_ME13 = 32,
    NUM_WIREGROUPS_ME21 = 112,
    NUM_WIREGROUPS_ME22 = 64,
    NUM_WIREGROUPS_ME31 = 96,
    NUM_WIREGROUPS_ME32 = 64,
    NUM_WIREGROUPS_ME41 = 96,
    NUM_WIREGROUPS_ME42 = 64,
    // this number should really be 112, but has always been 119 since the
    // CSC trigger was developed in 2006. Probably it would not hurt to change it to 112
    MAX_NUM_WIREGROUPS = 119
  };

  // distrips, strips, half-strips
  enum Strip_Info {
    // Each CFEB reads out 8 distrips...
    NUM_DISTRIPS_PER_CFEB = 8,
    //...16 strips...
    NUM_STRIPS_PER_CFEB = 2 * NUM_DISTRIPS_PER_CFEB,
    //...32 half-strips.
    NUM_HALF_STRIPS_PER_CFEB = 2 * NUM_STRIPS_PER_CFEB,
    // There are exactly 80 or 112 strips...
    MAX_NUM_STRIPS_RUN1 = MAX_CFEBS_RUN1 * NUM_STRIPS_PER_CFEB,  // 80
    MAX_NUM_STRIPS_RUN2 = MAX_CFEBS_RUN2 * NUM_STRIPS_PER_CFEB,  // 112
    //...and 160 or 224 half-strips for 5 or 7 CFEBs...
    MAX_NUM_HALF_STRIPS_RUN1 = MAX_CFEBS_RUN1 * NUM_HALF_STRIPS_PER_CFEB,  // 160
    MAX_NUM_HALF_STRIPS_RUN2 = MAX_CFEBS_RUN2 * NUM_HALF_STRIPS_PER_CFEB,  // 224
    // ...but depending on the chamber, there may or may not be strip staggering.
    /* CMS-MUO-16-001: "[..] alternate layers in a CSC are staggered by half a strip width, except
       in the ME1/1 chambers where the strips are narrower and the effect is small" */
    // _TRIGGER is added at the end, because these constants are only used in the trigger
    MAX_NUM_HALF_STRIPS_RUN1_TRIGGER = 1 + MAX_NUM_HALF_STRIPS_RUN1,  // 161
    MAX_NUM_HALF_STRIPS_RUN2_TRIGGER = 1 + MAX_NUM_HALF_STRIPS_RUN2,  // 225
    // Number of strips in ME11 (special case)
    NUM_STRIPS_ME1A_GANGED = NUM_CFEBS_ME1A_GANGED * NUM_STRIPS_PER_CFEB,      // 16
    NUM_STRIPS_ME1A_UNGANGED = NUM_CFEBS_ME1A_UNGANGED * NUM_STRIPS_PER_CFEB,  // 48
    NUM_STRIPS_ME1B = NUM_CFEBS_ME1B * NUM_STRIPS_PER_CFEB,                    // 64
    // Number of half-strips in ME11 (special case)
    NUM_HALF_STRIPS_ME1A_GANGED = NUM_CFEBS_ME1A_GANGED * NUM_HALF_STRIPS_PER_CFEB,      // 32
    NUM_HALF_STRIPS_ME1A_UNGANGED = NUM_CFEBS_ME1A_UNGANGED * NUM_HALF_STRIPS_PER_CFEB,  // 96
    NUM_HALF_STRIPS_ME1B = NUM_CFEBS_ME1B * NUM_HALF_STRIPS_PER_CFEB,                    // 128
    NUM_HALF_STRIPS_ME11_GANGED = NUM_CFEBS_ME11_GANGED * NUM_HALF_STRIPS_PER_CFEB,      // 160
    NUM_HALF_STRIPS_ME11_UNGANGED = NUM_CFEBS_ME11_UNGANGED * NUM_HALF_STRIPS_PER_CFEB,  // 224
    // max halfstrip number in ME1/1 chambers
    MAX_HALF_STRIP_ME1A_GANGED = NUM_HALF_STRIPS_ME1A_GANGED - 1,      // 31
    MAX_HALF_STRIP_ME1A_UNGANGED = NUM_HALF_STRIPS_ME1A_UNGANGED - 1,  // 95
    MAX_HALF_STRIP_ME1B = NUM_HALF_STRIPS_ME1B - 1,                    // 127
    // half-strips for the rest of the system
    NUM_HALF_STRIPS_ME12 = NUM_CFEBS_ME12 * NUM_HALF_STRIPS_PER_CFEB,  // 160
    NUM_HALF_STRIPS_ME13 = NUM_CFEBS_ME13 * NUM_HALF_STRIPS_PER_CFEB,  // 128
    NUM_HALF_STRIPS_ME21 = NUM_CFEBS_ME21 * NUM_HALF_STRIPS_PER_CFEB,  // 160
    NUM_HALF_STRIPS_ME22 = NUM_CFEBS_ME22 * NUM_HALF_STRIPS_PER_CFEB,  // 160
    NUM_HALF_STRIPS_ME31 = NUM_CFEBS_ME31 * NUM_HALF_STRIPS_PER_CFEB,  // 160
    NUM_HALF_STRIPS_ME32 = NUM_CFEBS_ME32 * NUM_HALF_STRIPS_PER_CFEB,  // 160
    NUM_HALF_STRIPS_ME41 = NUM_CFEBS_ME41 * NUM_HALF_STRIPS_PER_CFEB,  // 160
    NUM_HALF_STRIPS_ME42 = NUM_CFEBS_ME42 * NUM_HALF_STRIPS_PER_CFEB,  // 160
    // useful for the comparator code algorithm
    INVALID_HALF_STRIP = 65535
  };

  // CSCs have 6 layers. The key (reference) layer is the third layer
  enum Layer_Info { NUM_LAYERS = 6, KEY_CLCT_LAYER = 3, KEY_ALCT_LAYER = 3 };

  // Both ALCT and CLCTs have patterns. CLCTs have a better granularity than ALCTs, thus more patterns
  enum Pattern_Info {
    NUM_ALCT_PATTERNS = 3,
    ALCT_PATTERN_WIDTH = 5,
    // Run-1 and Run-2 CSC trigger patterns
    NUM_CLCT_PATTERNS = 11,
    // Run-3 CSC trigger patterns
    NUM_CLCT_PATTERNS_RUN3 = 5,
    CLCT_PATTERN_WIDTH = 11,
    // Max number of wires participating in a pattern
    MAX_WIRES_IN_PATTERN = 14,
    NUM_COMPARATOR_CODES = 4096
  };

  enum Digis_Info { MAX_DIGIS_PER_ALCT = 10, MAX_DIGIS_PER_CLCT = 8 };

  enum LCT_stubs {
    // CSC local trigger considers 4-bit BX window (16 numbers) in the readout
    MAX_CLCT_TBINS = 16,
    MAX_ALCT_TBINS = 16,
    MAX_LCT_TBINS = 16,
    // Maximum allowed matching window size
    MAX_MATCH_WINDOW_SIZE = 7,
    // Each CLCT processor can send up to 2 CLCTs to TMB per BX
    MAX_CLCTS_PER_PROCESSOR = 2,
    MAX_CLCTS_READOUT = 2,
    // Each ALCT processor can send up to 2 ALCTs to TMB per BX
    MAX_ALCTS_PER_PROCESSOR = 2,
    MAX_ALCTS_READOUT = 2,
    // Each CSC can send up to 2 LCTs to the MPC per BX
    MAX_LCTS_PER_CSC = 2,
    // An MPC receives up to 18 LCTs from 9 CSCs in the trigger sector
    MAX_LCTS_PER_MPC = 18,
    /*
      An EMTF sector processor receives LCTs from 5 MPCS
      or 45 chambers when not considering overlapping EMTF SPs
      18 CSCs in ME1; 9 x 3 CSCs in ME2,3,4
    */
    MAX_CSCS_PER_EMTF_SP_NO_OVERLAP = 45,
    // Reference BX for LCTs in simulation and firmware
    LCT_CENTRAL_BX = 8,
    /*
      Reference BX for ALCTs in firmware. In the ALCT simulation,
      and in the motherboard simulation the ALCT central BX is 8.
      However, ALCT BX is shifted before they are inserted into the EDM
      ROOT file to have a central BX of 3 and be consistent with the firmware.
     */
    ALCT_CENTRAL_BX = 3,
    /*
      Reference BX for CLCTs in firmware. In the CLCT simulation, the central
      CLCT BX is 7. In the motherboard simulation they are shifted to 8 (in order
      to matched with ALCTs). But in the EDM ROOT file the CLCT central BX is 7
      to be consistent with the firmware.
     */
    CLCT_CENTRAL_BX = 7,
    // Offset between the ALCT and CLCT central BX in simulation
    ALCT_CLCT_OFFSET = 1
  };
};

#endif
