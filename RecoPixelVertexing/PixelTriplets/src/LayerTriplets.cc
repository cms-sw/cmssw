#include "LayerTriplets.h"

using namespace ctfseeding;
std::vector<LayerTriplets::LayerPairAndLayers> LayerTriplets::layers() const
{
  std::vector<LayerPairAndLayers> result;
  typedef std::vector<LayerPairAndLayers>::iterator IR;

  typedef SeedingLayerSets::const_iterator IL;
  for (IL il=theSets.begin(), ilEnd= theSets.end(); il != ilEnd; ++il) {
    const SeedingLayers & set = *il;
    if (set.size() != 3) continue;
    SeedingLayerPair layerPair = std::make_pair(set[0], set[1]);
    bool added = false;
    for (IR ir = result.begin(); ir < result.end(); ++ir) {
      const SeedingLayerPair & resPair = ir->first;
      if (resPair.first ==layerPair.first && resPair.second == layerPair.second) {
        std::vector<SeedingLayer> & thirds = ir->second;
        thirds.push_back( set[2] );
        added = true;
        break;
      }
    }
    if (!added) {
      LayerPairAndLayers lpl = std::make_pair(layerPair,  std::vector<SeedingLayer>(1, set[2]) );
      result.push_back(lpl);
    }
  }
  return result;
}
