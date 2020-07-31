#ifndef L1Trigger_CSCTriggerPrimitives_CSCLCTTools_h
#define L1Trigger_CSCTriggerPrimitives_CSCLCTTools_h

#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <cmath>
#include <tuple>

namespace csctp {

  // CSC max strip & max wire
  unsigned get_csc_max_wire(int station, int ring);
  unsigned get_csc_max_halfstrip(int station, int ring);
  unsigned get_csc_max_quartstrip(int station, int ring);
  unsigned get_csc_max_eightstrip(int station, int ring);

  // CLCT min, max CFEB numbers
  std::pair<unsigned, unsigned> get_csc_min_max_cfeb(int station, int ring);

  // CSC min, max pattern
  std::pair<unsigned, unsigned> get_csc_min_max_pattern(bool isRun3);

  // CSC max quality
  unsigned get_csc_alct_max_quality(int station, int ring, bool isRun3);
  unsigned get_csc_clct_max_quality();
  unsigned get_csc_lct_max_quality();

}  // namespace csctp

#endif
