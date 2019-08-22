#ifndef DETECTOR_DESCRIPTION_DD_VECTOR_REGISTRY_H
#define DETECTOR_DESCRIPTION_DD_VECTOR_REGISTRY_H

#include <string>
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_vector.h"

namespace cms {
  struct DDVectorRegistry {
    tbb::concurrent_unordered_map<std::string, tbb::concurrent_vector<double> > vectors;
  };
}  // namespace cms

#endif
