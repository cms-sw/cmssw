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
  enum WG_and_Strip { MAX_NUM_WIRES = 119, MAX_NUM_STRIPS = 80,
			     NUM_DI_STRIPS = 40+1, // Add 1 to allow for staggering of strips
			     NUM_HALF_STRIPS = 160+1};

  enum Layer_Info { NUM_LAYERS = 6, KEY_CLCT_LAYER = 3, KEY_ALCT_LAYER = 3 };

  enum Pattern_Info { NUM_ALCT_PATTERNS = 3, NUM_CLCT_PATTERNS = 10 };

  enum Digis_Info { MAX_DIGIS_PER_ALCT = 10, MAX_DIGIS_PER_CLCT = 8 };

  enum MPC_stubs { maxStubs = 3 };

  /**
   * We assume that the digis which bx times differ from the current bx by
   * more than six will not contribute to the LCT rates at the current bx,
   * and ignore them.
   */

  enum Bx_Window { MIN_BUNCH = -6, MAX_BUNCH = 6, TOT_BUNCH = MAX_BUNCH - MIN_BUNCH + 1, TIME_OFFSET = -MIN_BUNCH };

};

#endif
