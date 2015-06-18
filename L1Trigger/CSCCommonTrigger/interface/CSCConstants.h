#ifndef CSCCommonTrigger_CSCConstants_h
#define CSCCommonTrigger_CSCConstants_h

/**
 * \class CSCConstants
 * \remark Port of ChamberConstants from ORCA
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

  enum Layer_Info { NUM_LAYERS = 6, KEY_CLCT_LAYER = 3, KEY_CLCT_LAYER_PRE_TMB07 = 4, KEY_ALCT_LAYER = 3 };

  enum Pattern_Info { NUM_ALCT_PATTERNS = 3, NUM_CLCT_PATTERNS = 11, NUM_CLCT_PATTERNS_PRE_TMB07 = 8 };

  enum Digis_Info { MAX_DIGIS_PER_ALCT = 10, MAX_DIGIS_PER_CLCT = 8 };

  enum MPC_stubs { maxStubs = 3 };

};

#endif
