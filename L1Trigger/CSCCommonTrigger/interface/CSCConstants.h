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
  enum WG_and_Strip { MAX_NUM_WIRES = 119, MAX_NUM_STRIPS = 80, MAX_NUM_STRIPS_7CFEBS = 112,
                      NUM_DI_STRIPS = 40+1, // Add 1 to allow for staggering of strips
                      NUM_HALF_STRIPS = 160+1, NUM_HALF_STRIPS_7CFEBS = 224+1};

  // CSCs have 6 layers. The key (refernce) layer is the third layer
  enum Layer_Info { NUM_LAYERS = 6, KEY_CLCT_LAYER = 3, KEY_CLCT_LAYER_PRE_TMB07 = 4, KEY_ALCT_LAYER = 3 };

  // Both ALCT and CLCTs have patterns. CLCTs have a better granularity than ALCTs, thus more patterns
  enum Pattern_Info { NUM_ALCT_PATTERNS = 3, NUM_CLCT_PATTERNS = 11, NUM_CLCT_PATTERNS_PRE_TMB07 = 8 };

  enum Digis_Info { MAX_DIGIS_PER_ALCT = 10, MAX_DIGIS_PER_CLCT = 8 };

  // Each CSC can send up to 2 LCTs to the MPC.
  // An MPC receives up to 18 LCTs from 9 CSCs in the trigger sector
  enum LCT_stubs{ MAX_LCTS_PER_CSC = 2, MAX_LCTS_PER_MPC = 18 };

};

#endif
