#ifndef LayerTriplets_H
#define LayerTriplets_H

/** A class grouping pixel layers in pairs and associating a vector
    of layers to each layer pair. The layer pair is used to generate
    hit pairs and the associated vector of layers to generate
    a third hit confirming layer pair
 */

#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

class LayerTriplets {
public:
  typedef std::pair<ctfseeding::SeedingLayer, ctfseeding::SeedingLayer> SeedingLayerPair;
  typedef std::pair<SeedingLayerPair, std::vector<ctfseeding::SeedingLayer> > LayerPairAndLayers;

  LayerTriplets( const ctfseeding::SeedingLayerSets & sets) : theSets(sets) {}

  std::vector<LayerPairAndLayers> layers() const;

private:
  ctfseeding::SeedingLayerSets theSets;
};

#endif

