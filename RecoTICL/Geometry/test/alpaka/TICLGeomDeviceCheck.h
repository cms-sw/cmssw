#ifndef RecoHGCal_TICL_test_alpaka_TICLGeomDeviceCheck_h
#define RecoHGCal_TICL_test_alpaka_TICLGeomDeviceCheck_h

#include <cstdint>

#include "CondFormats/HGCalObjects/interface/TICLGeomSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace ticlgeomtest {

    inline constexpr int kNMissingIds = 5;
    inline constexpr float kVertexZ = 1.5f;
    inline constexpr float kHitEnergy = 1.f;

    // Results of every ticlgeom::Cell getter for one cell.
    struct CellResult {
      int32_t index, indexAt;
      uint32_t rawDetId;
      ticlgeom::Position position;
      float eta, etaDisplaced, phi, pt, ptDisplaced;
      int32_t zside, layer, layerWithOffset, type, sensorGroup, cassette;
      bool masked, isSilicon, isScintillator, isBarrel;
      float thickness, radiusToSide;
      int32_t thicknessIndex;
      ticlgeom::CellCoord wafer, uv;
      bool isHalfCell;
      ticlgeom::WaferInfo waferInfo;
      ticlgeom::ScintDEtaDPhi dEtaDPhi;
      int32_t maxIphi;
      bool isFine;
      ticlgeom::TileInfo tileInfo;
    };

    // Results of the per-layer TICLGeomSoAConstView getters for one layer.
    struct LayerResult {
      float zPlus, zMinus, noseZ;
      ticlgeom::Position barrel;
    };

    // Results of the geometry-wide getters and of the invalid-input checks.
    struct ScalarResult {
      int32_t size, numberOfLayers, lastLayerEE, lastLayerFH, firstLayerBH, lastLayerBH, lastLayerECAL, lastLayerBarrel,
          noseLastLayer, maxWafersPerLayer, noseMaxWafersPerLayer, scintMaxIphi, geometryType;
      int32_t nBadInvalid;
    };

    struct GeomResults {
      int32_t nLayers;
      cms::alpakatools::host_buffer<CellResult[]> cells;
      cms::alpakatools::host_buffer<LayerResult[]> layers;
      cms::alpakatools::host_buffer<ScalarResult> scalars;
    };

    // Runs a kernel that looks up every cell by its rawDetId, both by binary
    // search (ticlgeom::indexOf) and through the lookup block
    // (ticlgeom::denseIdOf); returns the number of cells whose lookups do
    // not give back their own index.
    int32_t checkLookup(Queue& queue, TICLGeomCommonSoAConstView common, TICLGeomLookupSoAConstView lookup);

    // Runs every TICLGeomSoAConstView and ticlgeom::Cell getter on the device
    // for every cell, for every layer row and one layer past them, and returns
    // the results on the host.
    GeomResults runGeomQueries(Queue& queue, TICLGeomSoAConstView geom);

  }  // namespace ticlgeomtest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoHGCal_TICL_test_alpaka_TICLGeomDeviceCheck_h
