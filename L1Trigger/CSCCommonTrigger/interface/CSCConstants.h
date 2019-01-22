#ifndef CSCCommonTrigger_CSCConstants_h
#define CSCCommonTrigger_CSCConstants_h

/**
 * \class CSCConstants
 *
 * Static interface to basic chamber constants.
 */
#include <cmath>

class CSCConstants
{
 public:
  enum CFEB_Info {
    //Maximum number of cathode front-end boards
    MAX_CFEBS = 5,
  };

  enum FPGA_Latency{
    CLCT_EMUL_TIME_OFFSET = 3,
    ALCT_EMUL_TIME_OFFSET = 6
  };

  // Note: WIRE means actually "wiregroup" here
  enum WG_and_Strip {
    MAX_NUM_WIRES = 119,
    MAX_WIRES_ME11 = 48,
    MAX_NUM_STRIPS = 80,
    MAX_NUM_STRIPS_7CFEBS = 112,
    NUM_DI_STRIPS = 40+1, // Add 1 to allow for staggering of strips
    NUM_HALF_STRIPS = 160+1,
    NUM_HALF_STRIPS_7CFEBS = 224+1,
    // each CFEB reads out 8 distrips, 16 strips or 32 halfstrips
    NUM_DISTRIPS_PER_CFEB = 8,
    NUM_STRIPS_PER_CFEB = 16,
    NUM_HALF_STRIPS_PER_CFEB = 32,
    // max halfstrip number in ME1/1 chambers
    // All ME1A readout by 1 CFEB -> 32 -1
    MAX_HALF_STRIP_ME1A_GANGED = 31,
    // All ME1A readout by 3 CFEBs -> 3*32 -1
    MAX_HALF_STRIP_ME1A_UNGANGED = 95,
    // All ME1B readout by 4 CFEBs -> 4*32 -1
    MAX_HALF_STRIP_ME1B = 127,
    MAX_NUM_STRIPS_ME1B = 64,
    MAX_NUM_STRIPS_ME1A_GANGED = 16,
    MAX_NUM_STRIPS_ME1A_UNGANGED = 48,
  };

  // CSCs have 6 layers. The key (refernce) layer is the third layer
  enum Layer_Info {
    NUM_LAYERS = 6,
    KEY_CLCT_LAYER = 3,
    KEY_CLCT_LAYER_PRE_TMB07 = 4,
    KEY_ALCT_LAYER = 3 };

  // Both ALCT and CLCTs have patterns. CLCTs have a better granularity than ALCTs, thus more patterns
  enum Pattern_Info {
    NUM_ALCT_PATTERNS = 3,
    NUM_CLCT_PATTERNS = 11,
    NUM_CLCT_PATTERNS_PRE_TMB07 = 8,
    // Max number of wires participating in a pattern
    MAX_WIRES_IN_PATTERN = 14,
    // Max number of strips participating in a pattern
    MAX_STRIPS_IN_PATTERN = 26,
    // Max number of halfstrips participating in a pattern
    MAX_HALFSTRIPS_IN_PATTERN = 42};

  enum Digis_Info {
    MAX_DIGIS_PER_ALCT = 10,
    MAX_DIGIS_PER_CLCT = 8 };

  enum LCT_stubs{
    // CSC local trigger considers 4-bit BX window (16 numbers) in the readout
    MAX_CLCT_TBINS = 16,
    MAX_ALCT_TBINS = 16,
    MAX_LCT_TBINS = 16,
    // Maximum allowed matching window size
    MAX_MATCH_WINDOW_SIZE = 15,
    // Each CLCT processor can send up to 2 CLCTs to TMB per BX
    MAX_CLCTS_PER_PROCESSOR = 2,
    // Each ALCT processor can send up to 2 ALCTs to TMB per BX
    MAX_ALCTS_PER_PROCESSOR = 2,
    // Each CSC can send up to 2 LCTs to the MPC per BX
    MAX_LCTS_PER_CSC = 2,
    // An MPC receives up to 18 LCTs from 9 CSCs in the trigger sector
    MAX_LCTS_PER_MPC = 18,
    // Reference BX for LCTs in simulation and firmware
    LCT_CENTRAL_BX = 8};
};

#endif
