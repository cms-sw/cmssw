#ifndef L1Trigger_CSCTriggerPrimitives_CSCPatternBank_h
#define L1Trigger_CSCTriggerPrimitives_CSCPatternBank_h

#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include <vector>

//
// Class with only static members that contains the ALCT and CLCT trigger patterns
//

class CSCPatternBank {
public:
  typedef std::vector<std::vector<std::vector<int> > > CLCTPatterns;

  typedef int ALCTPatterns[CSCConstants::NUM_ALCT_PATTERNS][CSCConstants::MAX_WIRES_IN_PATTERN];

  /** Pre-defined ALCT patterns. */

  // This is the pattern envelope, which is used to define the collision
  // patterns A and B.
  static const int alct_pattern_envelope[CSCConstants::MAX_WIRES_IN_PATTERN];

  // key wire offsets for ME1 and ME2 are the same
  // offsets for ME3 and ME4 are the same
  static const int alct_keywire_offset[2][CSCConstants::MAX_WIRES_IN_PATTERN];

  // Since the test beams in 2003, both collision patterns are "completely
  // open".  This is our current default.
  static const int alct_pattern_mask_open[CSCConstants::NUM_ALCT_PATTERNS][CSCConstants::MAX_WIRES_IN_PATTERN];

  // Special option for narrow pattern for ring 1 stations
  static const int alct_pattern_mask_r1[CSCConstants::NUM_ALCT_PATTERNS][CSCConstants::MAX_WIRES_IN_PATTERN];

  /** Pre-defined CLCT patterns. */

  // New set of halfstrip patterns for 2007 version of the algorithm.
  // For the given pattern, set the unused parts of the pattern to 999.
  // Pattern[i][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN] contains bend direction.
  // Bend of 0 is right/straight and bend of 1 is left.
  // Pattern[i][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN+1] contains pattern maximum width
  static const int clct_pattern[CSCConstants::NUM_CLCT_PATTERNS][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN + 2];

  // New patterns for Run-3
  static const CLCTPatterns clct_pattern_run3_;

  // half strip offsets per layer for each half strip in the pattern envelope
  static const int clct_pattern_offset[CSCConstants::MAX_HALFSTRIPS_IN_PATTERN];
};

#endif
