#ifndef DETECTOR_DESCRIPTION_DD_VECTOR_REGISTRY_H
#define DETECTOR_DESCRIPTION_DD_VECTOR_REGISTRY_H

#include <string>
#include <vector>
#include <unordered_map>

namespace cms {
  struct DDVectorRegistry {
    std::unordered_map< std::string, std::vector<double> > vectors;
  };
}

#endif
