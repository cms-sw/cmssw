#ifndef LayerQuadruplets_H
#define LayerQuadruplets_H

/** A class grouping pixel layers in triplets and associating a vector
    of layers to each layer pair. The layer triplet is used to generate
    hit triplets and the associated vector of layers to generate
    a fourth hit confirming layer triplet
 */

#include <vector>
#include <tuple>
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

namespace LayerQuadruplets {
  using Layer = SeedingLayerSetsHits::SeedingLayer;
  using LayerSet = SeedingLayerSetsHits::SeedingLayerSet;
  using LayerSetAndLayers = std::pair<LayerSet, std::vector<Layer> >;

  std::vector<LayerSetAndLayers> layers(const SeedingLayerSetsHits& sets);
};

#endif

