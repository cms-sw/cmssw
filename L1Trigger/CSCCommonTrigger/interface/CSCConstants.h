#ifndef L1Trigger_CSCConstants_h
#define L1Trigger_CSCConstants_h

/**
 * \class CSCConstants
 * \remark Port of ChamberConstants from ORCA
 *
 * Static interface to basic chamber constants.
 */
#include <L1Trigger/CSCCommonTrigger/interface/CSCBitWidths.h>
#include <cmath>

class CSCConstants
{
 public:
  static enum WG_and_Strip { MAX_NUM_WIRES = 119, MAX_NUM_STRIPS = 80,
			     NUM_DI_STRIPS = 40+1 // Add 1 to allow for staggering of strips
			     NUM_HALF_STRIPS = 160+1};

  static enum Layer_Info { NUM_LAYERS = 6, KEY_LAYER = 4 }; // shouldn't key layer be 3?

  static enum Pattern_Info { NUM_ALCT_PATTERNS = 3, NUM_CLCT_PATTERNS = 8
			     MAX_CLCT_PATTERNS = 1<<CSCBitWidths::CLCT_PATTERN_BITS };

  static enum Digis_Info { MAX_DIGIS_PER_ALCT = 10, MAX_DIGIS_PER_CLCT = 8 };

  /**
   * We assume that the digis which bx times differ from the current bx by
   * more than six will not contribute to the LCT rates at the current bx,
   * and ignore them.
   */

  static enum Bx_Window { MIN_BUNCH = -6, MAX_BUNCH = 6, TOT_BUNCH = MAX_BUNCH - MIN_BUNCH + 1, TIME_OFFSET = std::abs(MIN_BUNCH)};

  static const double RAD_PER_DEGREE = M_PI/180.; // where to get PI from?
  
  /// The center of the first "perfect" sector in phi.
  static const double SECTOR1_CENT_DEG = 45.;
  static const double SECTOR1_CENT_RAD = SECTOR1_CENT_DEG * RAD_PER_DEGREE;
  
  /** 
   * Sector size is 62 degrees.  Nowadays (in ORCA6) the largest size
   * of ideal sectors is 61.37 degrees (it is more than 60 because of
   * overlaps between sectors), but we leave some more space to handle
   * movements of the disks of about 8 mm.
   */  
  static const double SECTOR_DEG = 62.;
  static const double SECTOR_RAD = SECTOR_DEG*RAD_PER_DEGREE; // radians
  // needs BX info and some special station 1 info
};

#endif
