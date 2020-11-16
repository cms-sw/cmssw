#ifndef DETECTOR_DESCRIPTION_DD_VECTOR_REGISTRY_H
#define DETECTOR_DESCRIPTION_DD_VECTOR_REGISTRY_H

#include <string>
#include <unordered_map>
#include <vector>

namespace cms {
  using DDVectorsMap = std::unordered_map<std::string, std::vector<double>>;

  struct DDVectorRegistry {
    DDVectorsMap vectors;
  };
}  // namespace cms

#endif
