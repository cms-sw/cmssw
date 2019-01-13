#ifndef DETECTOR_DESCRIPTION_DD_SPECPAR_REGISTRY_H
#define DETECTOR_DESCRIPTION_DD_SPECPAR_REGISTRY_H

#include <string>
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_vector.h"

namespace cms {
  using DDVectorsMap = tbb::concurrent_unordered_map< std::string, tbb::concurrent_vector<double>>;
  using DDPartSelectionMap = tbb::concurrent_unordered_map< std::string, tbb::concurrent_vector<std::string>>;
  
  struct DDSpecPar {
    tbb::concurrent_vector<std::string> paths;
    DDPartSelectionMap spars;
    DDVectorsMap numpars;
  };

  struct DDSpecParRegistry {
    tbb::concurrent_unordered_map<std::string, DDSpecPar> specpars;
  };
}

#endif
