#ifndef DETECTOR_DESCRIPTION_MUON_NUMBERING_H
#define DETECTOR_DESCRIPTION_MUON_NUMBERING_H

#include <string>
#include "tbb/concurrent_unordered_map.h"

namespace cms {

  struct MuonNumbering {
    tbb::concurrent_unordered_map<std::string, int> values;
  };
}

#endif
