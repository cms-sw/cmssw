#ifndef DETECTOR_DESCRIPTION_DD_SPECPAR_REGISTRY_H
#define DETECTOR_DESCRIPTION_DD_SPECPAR_REGISTRY_H

#include <string>
#include <string_view>
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_vector.h"

namespace cms {
  using DDPaths = tbb::concurrent_vector<std::string>;
  using DDPartSelectionMap = tbb::concurrent_unordered_map< std::string, tbb::concurrent_vector<std::string>>;
  using DDVectorsMap = tbb::concurrent_unordered_map< std::string, tbb::concurrent_vector<double>>;
  
  struct DDSpecPar {
    std::vector<std::string_view> tails(const std::vector<std::string_view>&) const;
    std::vector<std::string_view> split(std::string_view, const char*) const;
    std::vector<std::string_view> vPathsTo(unsigned int) const;
    std::string_view realTopName(std::string_view input) const;
    std::string_view strValue(const char*) const;
    double dblValue(const char*) const;
    
    DDPaths paths;
    DDPartSelectionMap spars;
    DDVectorsMap numpars;
  };
  
  using DDSpecParMap = tbb::concurrent_unordered_map<std::string, DDSpecPar>;
  using DDSpecParRefMap = tbb::concurrent_unordered_map<const std::string*, const DDSpecPar*>;

  struct DDSpecParRegistry {
    void filter(DDSpecParRefMap&, std::string_view, std::string_view) const;
    
    DDSpecParMap specpars;
  };
}

#endif
