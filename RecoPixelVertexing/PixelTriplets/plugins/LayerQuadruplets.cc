#include "LayerQuadruplets.h"

namespace LayerQuadruplets {
std::vector<LayerSetAndLayers> layers(const SeedingLayerSetsHits& sets) {
  std::vector<LayerSetAndLayers> result;
  if(sets.numberOfLayersInSet() != 4)
    return result;

  for(LayerSet set: sets) {
    bool added = false;

    for(auto ir = result.begin(); ir < result.end(); ++ir) {
      const LayerSet & resTriplet = ir->first;
      if(resTriplet[0].index() == set[0].index() &&
         resTriplet[1].index() == set[1].index() &&
         resTriplet[2].index() == set[2].index()) {
        std::vector<Layer>& fourths = ir->second;
        fourths.push_back( set[3] );
        added = true;
        break;
      }
    }
    if (!added) {
      LayerSetAndLayers ltl = std::make_pair(set, std::vector<Layer>(1, set[3]) );
      result.push_back(ltl);
    }
  }
  return result;
}
}
