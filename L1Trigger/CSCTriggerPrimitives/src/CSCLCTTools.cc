#include "L1Trigger/CSCTriggerPrimitives/interface/CSCLCTTools.h"

namespace csctp {

  // Number of halfstrips and wiregroups
  // +----------------------------+------------+------------+
  // | Chamber type               | Num of     | Num of     |
  // |                            | halfstrips | wiregroups |
  // +----------------------------+------------+------------+
  // | ME1/1a                     | 96         | 48         |
  // | ME1/1b                     | 128        | 48         |
  // | ME1/2                      | 160        | 64         |
  // | ME1/3                      | 128        | 32         |
  // | ME2/1                      | 160        | 112        |
  // | ME3/1, ME4/1               | 160        | 96         |
  // | ME2/2, ME3/2, ME4/2        | 160        | 64         |
  // +----------------------------+------------+------------+

  unsigned get_csc_max_wire(int station, int ring) {
    unsigned max_wire = 0;            // wiregroup
    if (station == 1 && ring == 4) {  // ME1/1a
      max_wire = 48;
    } else if (station == 1 && ring == 1) {  // ME1/1b
      max_wire = 48;
    } else if (station == 1 && ring == 2) {  // ME1/2
      max_wire = 64;
    } else if (station == 1 && ring == 3) {  // ME1/3
      max_wire = 32;
    } else if (station == 2 && ring == 1) {  // ME2/1
      max_wire = 112;
    } else if (station >= 3 && ring == 1) {  // ME3/1, ME4/1
      max_wire = 96;
    } else if (station >= 2 && ring == 2) {  // ME2/2, ME3/2, ME4/2
      max_wire = 64;
    }
    return max_wire;
  }

  unsigned get_csc_max_halfstrip(int station, int ring) {
    int max_strip = 0;                // halfstrip
    if (station == 1 && ring == 4) {  // ME1/1a
      max_strip = 96;
    } else if (station == 1 && ring == 1) {  // ME1/1b
      // In the CSC local trigger
      // ME1/a is taken together with ME1/b
      max_strip = 128 + 96;
    } else if (station == 1 && ring == 2) {  // ME1/2
      max_strip = 160;
    } else if (station == 1 && ring == 3) {  // ME1/3
      max_strip = 128;
    } else if (station == 2 && ring == 1) {  // ME2/1
      max_strip = 160;
    } else if (station >= 3 && ring == 1) {  // ME3/1, ME4/1
      max_strip = 160;
    } else if (station >= 2 && ring == 2) {  // ME2/2, ME3/2, ME4/2
      max_strip = 160;
    }
    return max_strip;
  }

  unsigned get_csc_max_quartstrip(int station, int ring) { return get_csc_max_halfstrip(station, ring) * 2; }

  unsigned get_csc_max_eightstrip(int station, int ring) { return get_csc_max_halfstrip(station, ring) * 4; }

  std::pair<unsigned, unsigned> get_csc_min_max_cfeb(int station, int ring) {
    // 5 CFEBs [0,4] for non-ME1/1 chambers
    int min_cfeb = 0;
    int max_cfeb = CSCConstants::MAX_CFEBS - 1;  // 4
    // 7 CFEBs [0,6] for ME1/1 chambers
    if (station == 1 and ring == 1) {
      max_cfeb = 6;
    }
    return std::make_pair(min_cfeb, max_cfeb);
  }

  std::pair<unsigned, unsigned> get_csc_min_max_pattern(bool isRun3) {
    int min_pattern, max_pattern;
    // Run-1 or Run-2 case
    if (!isRun3) {
      min_pattern = 2;
      max_pattern = 10;
      // Run-3 case
    } else {
      min_pattern = 0;
      max_pattern = 4;
    }
    return std::make_pair(min_pattern, max_pattern);
  }

  unsigned get_csc_alct_max_quality(int station, int ring, bool runGEMCSC) {
    int max_quality = 3;
    // GE2/1-ME2/1 ALCTs are allowed 3-layer ALCTs
    if (runGEMCSC and station == 2 and ring == 1) {
      max_quality = 4;
    }
    return max_quality;
  }

  unsigned get_csc_clct_max_quality() {
    int max_quality = 6;
    return max_quality;
  }

  unsigned get_csc_lct_max_quality() {
    int max_quality = 15;
    return max_quality;
  }

}  // namespace csctp
