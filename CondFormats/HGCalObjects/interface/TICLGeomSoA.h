#ifndef CondFormats_HGCalObjects_interface_TICLGeomSoA_h
#define CondFormats_HGCalObjects_interface_TICLGeomSoA_h

#include <cmath>
#include <cstdint>
#include <limits>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Per-cell replacement of the hgcal::RecHitTools detid-keyed methods,
// usable on host and device, split into three SoABlocks so cell-type
// specific columns are allocated only for the cells that have them. The
// cells are rawDetId-ordered, which groups them as [barrel | silicon | scint]
// (detector ids sort Ecal < Hcal < HFNose < HGCalEE < HGCalHSi < HGCalHSc,
// and only HGCalHSc is scintillator), so each range is contiguous and the
// silicon / scint block-local index of global row i is arithmetic:
// i - nBarrel for silicon, i - nBarrel - nSilicon for scint.
//
// common (all cells):
//   x, y, z                  getPosition (eta/phi derived from x,y,z on read)
//   zside                    zside (+/-1, 0 where RecHitTools returns 0)
//   layer, layerWithOffset   getLayer, getLayerWithOffset
//   cellType, sensorGroup    getCellType, getSensorGroup (-1 / UNKNOWN
//                            outside HGCal and HFNose)
//   cassette                 getWaferInfo / getTileInfo cassette (shared)
//   isSilicon, isScintillator, isBarrel, masked
//   scalars                  the geometry-wide getters, plus nBarrel and
//                            nSilicon for block-local indexing
// silicon (silicon cells only):
//   siThickness, radiusToSide, waferU, waferV, cellU, cellV,
//   waferPartialType, waferOrientation, waferPlacementIndex, waferType,
//   siThickIndex, isHalfCell
// scint (scintillator cells only):
//   scintDEta, scintDPhi, scintMaxIphi, tileType, tileSipm, isScintillatorFine
GENERATE_SOA_LAYOUT(TICLGeomCommonSoALayout,
                    SOA_COLUMN(uint32_t, rawDetId),
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(int16_t, layer),
                    SOA_COLUMN(int16_t, layerWithOffset),
                    SOA_COLUMN(int16_t, cassette),
                    SOA_COLUMN(int8_t, zside),
                    SOA_COLUMN(int8_t, cellType),
                    SOA_COLUMN(int8_t, sensorGroup),
                    SOA_COLUMN(bool, isSilicon),
                    SOA_COLUMN(bool, isScintillator),
                    SOA_COLUMN(bool, isBarrel),
                    SOA_COLUMN(bool, masked),
                    SOA_SCALAR(int32_t, lastLayerEE),
                    SOA_SCALAR(int32_t, lastLayerFH),
                    SOA_SCALAR(int32_t, firstLayerBH),
                    SOA_SCALAR(int32_t, lastLayerBH),
                    SOA_SCALAR(int32_t, numberOfLayers),
                    SOA_SCALAR(int32_t, lastLayerECAL),
                    SOA_SCALAR(int32_t, lastLayerBarrel),
                    SOA_SCALAR(int32_t, maxNumberOfWafersPerLayer),
                    SOA_SCALAR(int32_t, bhMaxIphi),
                    SOA_SCALAR(int32_t, geometryType),
                    SOA_SCALAR(int32_t, noseLastLayer),
                    SOA_SCALAR(int32_t, maxNumberOfWafersNose),
                    SOA_SCALAR(int32_t, nBarrel),
                    SOA_SCALAR(int32_t, nSilicon))

GENERATE_SOA_LAYOUT(TICLGeomSiliconSoALayout,
                    SOA_COLUMN(float, siThickness),
                    SOA_COLUMN(float, radiusToSide),
                    SOA_COLUMN(int16_t, waferU),
                    SOA_COLUMN(int16_t, waferV),
                    SOA_COLUMN(int16_t, cellU),
                    SOA_COLUMN(int16_t, cellV),
                    SOA_COLUMN(int16_t, waferPartialType),
                    SOA_COLUMN(int16_t, waferOrientation),
                    SOA_COLUMN(int16_t, waferPlacementIndex),
                    SOA_COLUMN(int8_t, waferType),
                    SOA_COLUMN(int8_t, siThickIndex),
                    SOA_COLUMN(bool, isHalfCell))

GENERATE_SOA_LAYOUT(TICLGeomScintSoALayout,
                    SOA_COLUMN(float, scintDEta),
                    SOA_COLUMN(float, scintDPhi),
                    SOA_COLUMN(int16_t, scintMaxIphi),
                    SOA_COLUMN(int8_t, tileType),
                    SOA_COLUMN(int8_t, tileSipm),
                    SOA_COLUMN(bool, isScintillatorFine))

GENERATE_SOA_BLOCKS(TICLGeomSoALayout,
                    SOA_BLOCK(common, TICLGeomCommonSoALayout),
                    SOA_BLOCK(silicon, TICLGeomSiliconSoALayout),
                    SOA_BLOCK(scint, TICLGeomScintSoALayout))

using TICLGeomCommonSoA = TICLGeomCommonSoALayout<>;
using TICLGeomCommonSoAConstView = TICLGeomCommonSoA::ConstView;
using TICLGeomSiliconSoA = TICLGeomSiliconSoALayout<>;
using TICLGeomSiliconSoAConstView = TICLGeomSiliconSoA::ConstView;
using TICLGeomScintSoA = TICLGeomScintSoALayout<>;
using TICLGeomScintSoAConstView = TICLGeomScintSoA::ConstView;

using TICLGeomSoA = TICLGeomSoALayout<>;
using TICLGeomSoAView = TICLGeomSoA::View;
using TICLGeomSoAConstView = TICLGeomSoA::ConstView;

namespace ticlgeom {

  // Sentinel for wafer and cell coordinates of cells that have none
  // (RecHitTools returns int max there; the SoA stores int16).
  constexpr int16_t kInvalidCoord = std::numeric_limits<int16_t>::max();

  // Cells are stored ordered by increasing rawDetId, so the index of a detid
  // is found by binary search over the common block, on host and device
  // alike. Returns -1 when the detid is not part of the geometry.
  SOA_HOST_DEVICE SOA_INLINE int32_t indexOf(TICLGeomCommonSoAConstView const& common, uint32_t rawDetId) {
    int32_t lo = 0;
    int32_t hi = common.metadata().size() - 1;
    while (lo <= hi) {
      int32_t mid = lo + (hi - lo) / 2;
      uint32_t val = common[mid].rawDetId();
      if (val == rawDetId)
        return mid;
      if (val < rawDetId)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    return -1;
  }

  // RecHitTools::getEta with a displaced vertex; with vertex_z = 0 it equals
  // the eta derived from the stored position.
  SOA_HOST_DEVICE SOA_INLINE float etaFromVertex(TICLGeomCommonSoAConstView const& common, int32_t i, float vertex_z) {
    const float dz = common[i].z() - vertex_z;
    const float rho = std::sqrt(common[i].x() * common[i].x() + common[i].y() * common[i].y());
    return std::asinh(dz / rho);
  }

  // RecHitTools::getPt
  SOA_HOST_DEVICE SOA_INLINE float pt(TICLGeomCommonSoAConstView const& common,
                                      int32_t i,
                                      float hitEnergy,
                                      float vertex_z) {
    return hitEnergy / std::cosh(etaFromVertex(common, i, vertex_z));
  }

  // RecHitTools::isOnlySilicon, from the layer scalars
  SOA_HOST_DEVICE SOA_INLINE bool isOnlySilicon(TICLGeomCommonSoAConstView const& common, int32_t layerWithOffset) {
    return layerWithOffset < common.firstLayerBH();
  }

}  // namespace ticlgeom

#endif  // CondFormats_HGCalObjects_interface_TICLGeomSoA_h
