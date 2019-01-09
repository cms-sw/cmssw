#ifndef DETECTOR_DESCRIPTION_DD_SPECPAR_REGISTRY_H
#define DETECTOR_DESCRIPTION_DD_SPECPAR_REGISTRY_H

#include <string>
#include <vector>
#include <unordered_map>

namespace cms {
  using DDVectorsMap = std::unordered_map< std::string, std::vector<double>>;
  using DDPartSelectionMap = std::unordered_map< std::string, std::vector<std::string>>;
  
  struct DDSpecPar {
    std::vector<std::string> paths;
    DDPartSelectionMap spars;
    DDVectorsMap numpars;
  };

  struct DDSpecParRegistry {
    std::unordered_map<std::string, DDSpecPar> specpars;
  };
}

#endif
