#ifndef DataFormats_HGCalReco_Common_h
#define DataFormats_HGCalReco_Common_h

#include <vector>
#include <array>
#include <cstdint>

namespace ticl::constants {
  constexpr float minEta = 1.5f;
  constexpr float maxEta = 3.2f;
  constexpr int nEtaBins = 34;
  constexpr int nPhiBins = 126;
  constexpr int nLayers = 104;
}  // namespace ticl::constants

class TICLLayerTile;
namespace ticl {
  typedef std::vector<std::pair<unsigned int, float> > HgcalClusterFilterMask;
  typedef std::array<std::vector<unsigned int>, constants::nEtaBins * constants::nPhiBins> Tile;
  typedef std::array<TICLLayerTile, ticl::constants::nLayers> Tiles;
}  // namespace ticl


namespace ticlHFNose::constants {
  constexpr float minEta = 3.f;
  constexpr float maxEta = 4.2f;
  constexpr int nEtaBins = 34;
  constexpr int nPhiBins = 126;
  constexpr int nLayers = 16;
}  // namespace ticlHFnose::constants

class TICLHFNoseLayerTile;
namespace ticlHFNose {
  typedef std::vector<std::pair<unsigned int, float> > HgcalClusterFilterMask;
  typedef std::array<std::vector<unsigned int>, constants::nEtaBins * constants::nPhiBins> Tile;
  typedef std::array<TICLHFNoseLayerTile, ticlHFNose::constants::nLayers> Tiles;
}  // namespace ticlHFnose

#endif  // DataFormats_HGCalReco_Common_h
