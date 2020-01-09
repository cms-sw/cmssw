#ifndef DETECTOR_DESCRIPTION_DD_SPECPAR_REGISTRY_H
#define DETECTOR_DESCRIPTION_DD_SPECPAR_REGISTRY_H

#include <string>
#include <string_view>
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_vector.h"

namespace cms {
  using DDPaths = tbb::concurrent_vector<std::string>;
  using DDPartSelectionMap = tbb::concurrent_unordered_map<std::string, tbb::concurrent_vector<std::string>>;
  using DDVectorsMap = tbb::concurrent_unordered_map<std::string, tbb::concurrent_vector<double>>;

  struct DDSpecPar {
    std::string_view strValue(const std::string&) const;
    bool hasValue(const std::string& key) const;
    double dblValue(const std::string&) const;

    template <typename T>
    T value(const std::string&) const;

    DDPaths paths;
    DDPartSelectionMap spars;
    DDVectorsMap numpars;
  };

  using DDSpecParMap = tbb::concurrent_unordered_map<std::string, DDSpecPar>;
  using DDSpecParRefs = std::vector<const DDSpecPar*>;

  struct DDSpecParRegistry {
    void filter(DDSpecParRefs&, std::string_view, std::string_view) const;
    void filter(DDSpecParRefs&, std::string_view) const;
    std::vector<std::string_view> names() const;
    bool hasSpecPar(std::string_view) const;
    const DDSpecPar* specPar(std::string_view) const;

    DDSpecParMap specpars;
  };
}  // namespace cms

#endif
