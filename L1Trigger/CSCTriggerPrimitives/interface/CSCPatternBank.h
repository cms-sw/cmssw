#ifndef L1Trigger_CSCTriggerPrimitives_CSCPatternBank_h
#define L1Trigger_CSCTriggerPrimitives_CSCPatternBank_h

#include "DataFormats/L1TMuon/interface/CSCConstants.h"
#include <vector>

//
// Class with only static members that contains the ALCT and CLCT trigger patterns
//

class CSCPatternBank {
public:
  // typedef used for both ALCT and CLCT
  typedef std::vector<std::vector<std::vector<int> > > LCTPatterns;

  /** Pre-defined ALCT patterns. */

  // key wire offsets for ME1 and ME2 are the same
  // offsets for ME3 and ME4 are the same
  static const int alct_keywire_offset_[2][CSCConstants::ALCT_PATTERN_WIDTH];

  // Since the test beams in 2003, both collision patterns are "completely
  // open".  This is our current default.
  static const LCTPatterns alct_pattern_legacy_;

  // Special option for narrow pattern for ring 1 stations
  static const LCTPatterns alct_pattern_r1_;

  /** Pre-defined CLCT patterns. */

  // New set of halfstrip patterns for 2007 version of the algorithm.
  // For the given pattern, set the unused parts of the pattern to 0.
  // The bend direction is given by the next-to-last number in the the 6th layer
  // Bend of 0 is right/straight and bend of 1 is left.
  // The pattern maximum width is the last number in the the 6th layer
  // Use during Run-1 and Run-2
  static const LCTPatterns clct_pattern_legacy_;

  // New patterns for Run-3
  static const LCTPatterns clct_pattern_run3_;

  // half strip offsets per layer for each half strip in the pattern envelope
  static const int clct_pattern_offset_[CSCConstants::CLCT_PATTERN_WIDTH];
};

#endif
