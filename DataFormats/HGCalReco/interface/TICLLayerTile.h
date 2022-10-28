// Authors: Marco Rovere, Felice Pantaleo - marco.rovere@cern.ch, felice.pantaleo@cern.ch
// Date: 05/2019

#ifndef DataFormats_HGCalReco_TICLLayerTile_h
#define DataFormats_HGCalReco_TICLLayerTile_h

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

template <typename T>
class TICLLayerTileT {
public:
  typedef T type;

  void fill(double eta, double phi, unsigned int layerClusterId) {
    tile_[globalBin(eta, phi)].push_back(layerClusterId);
  }

  int etaBin(float eta) const {
    constexpr float etaRange = T::maxEta - T::minEta;
    static_assert(etaRange >= 0.f);
    float r = T::nEtaBins / etaRange;
    int etaBin = (std::abs(eta) - T::minEta) * r;
    etaBin = std::clamp(etaBin, 0, T::nEtaBins - 1);
    return etaBin;
  }

  int phiBin(float phi) const {
    auto normPhi = normalizedPhi(phi);
    float r = T::nPhiBins * M_1_PI * 0.5f;
    int phiBin = (normPhi + M_PI) * r;

    return phiBin;
  }

  std::array<int, 4> searchBoxEtaPhi(float etaMin, float etaMax, float phiMin, float phiMax) const {
    // The tile only handles one endcap at a time and does not hold mixed eta
    // values.
    if (etaMin * etaMax < 0) {
      return std::array<int, 4>({{0, 0, 0, 0}});
    }
    if (etaMax - etaMin < 0) {
      return std::array<int, 4>({{0, 0, 0, 0}});
    }
    int etaBinMin = etaBin(etaMin);
    int etaBinMax = etaBin(etaMax);
    int phiBinMin = phiBin(phiMin);
    int phiBinMax = phiBin(phiMax);
    if (etaMin < 0) {
      std::swap(etaBinMin, etaBinMax);
    }
    // If the search window cross the phi-bin boundary, add T::nPhiBins to the
    // MAx value. This guarantees that the caller can perform a valid doule
    // loop on eta and phi. It is the caller responsibility to perform a module
    // operation on the phiBin values returned by this function, to explore the
    // correct bins.
    if (phiBinMax < phiBinMin) {
      phiBinMax += T::nPhiBins;
    }
    return std::array<int, 4>({{etaBinMin, etaBinMax, phiBinMin, phiBinMax}});
  }

  int globalBin(int etaBin, int phiBin) const { return phiBin + etaBin * T::nPhiBins; }

  int globalBin(double eta, double phi) const { return phiBin(phi) + etaBin(eta) * T::nPhiBins; }

  void clear() {
    auto nBins = T::nEtaBins * T::nPhiBins;
    for (int j = 0; j < nBins; ++j)
      tile_[j].clear();
  }

  const std::vector<unsigned int>& operator[](int globalBinId) const { return tile_[globalBinId]; }

private:
  std::array<std::vector<unsigned int>, T::nBins> tile_;
};

namespace ticl {
  using TICLLayerTile = TICLLayerTileT<TileConstants>;
  using Tiles = std::array<TICLLayerTile, TileConstants::nLayers>;
  using TracksterTiles = std::array<TICLLayerTile, TileConstants::iterations>;

  using TICLLayerTileHFNose = TICLLayerTileT<TileConstantsHFNose>;
  using TilesHFNose = std::array<TICLLayerTileHFNose, TileConstantsHFNose::nLayers>;
  using TracksterTilesHFNose = std::array<TICLLayerTileHFNose, TileConstantsHFNose::iterations>;

}  // namespace ticl

template <typename T>
class TICLGenericTile {
public:
  // value_type_t is the type of the type of the array used by the incoming <T> type.
  using constants_type_t = typename T::value_type::type;
  // This class represents a generic collection of Tiles. The additional index
  // numbering is not handled internally. It is the user's responsibility to
  // properly use and consistently access it here.
  const auto& operator[](int index) const { return tiles_[index]; }
  void fill(int index, double eta, double phi, unsigned int objectId) { tiles_[index].fill(eta, phi, objectId); }

private:
  T tiles_;
};

using TICLLayerTiles = TICLGenericTile<ticl::Tiles>;
using TICLTracksterTiles = TICLGenericTile<ticl::TracksterTiles>;
using TICLLayerTilesHFNose = TICLGenericTile<ticl::TilesHFNose>;
using TICLTracksterTilesHFNose = TICLGenericTile<ticl::TracksterTilesHFNose>;

#endif
