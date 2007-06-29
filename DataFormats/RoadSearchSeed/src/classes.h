#ifndef ROADSEARCHSEED_CLASSES_H
#define ROADSEARCHSEED_CLASSES_H

#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<RoadSearchSeedCollection> roadSearchSeedCollectionWrapper;
    edm::Ref<std::vector<RoadSearchSeed>, RoadSearchSeed> roadSearchSeedRef;
    edm::RefVector<std::vector<RoadSearchSeed>, RoadSearchSeed> roadSearchSeedRefVector;

  }
}

#endif // ROADSEARCHSEED_CLASSES_H
