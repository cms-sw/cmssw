#ifndef CondFormats_HGCalObjects_interface_TICLGeomSoA_h
#define CondFormats_HGCalObjects_interface_TICLGeomSoA_h

#include <cstdint>
#include <limits>
#include <type_traits>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include <xtd/math/asinh.h>
#include <xtd/math/atan2.h>
#include <xtd/math/cosh.h>
#include <xtd/math/sqrt.h>

// Per-cell replacement of the hgcal::RecHitTools detid-keyed methods,
// usable on host and device. Cell-type specific columns are in separate
// blocks, so they are allocated only for the cells that have them. The
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

// Coarse-block lookup table from a detid to its row in the common block.
// Row ticlgeom::coarseSlot(detid) holds the start row and the length of the
// contiguous run of cells that share the detid bits above the cell
// coordinates (one silicon wafer or one scintillator ring). One table per
// HGCal subdetector (EE, HSi, HSc), concatenated at the base offsets. The
// block is empty for a collection without HGCal cells.
GENERATE_SOA_LAYOUT(TICLGeomLookupSoALayout,
                    SOA_COLUMN(int32_t, blockStart),
                    SOA_COLUMN(int32_t, blockCount),
                    SOA_SCALAR(int32_t, eeBase),
                    SOA_SCALAR(int32_t, hsiBase),
                    SOA_SCALAR(int32_t, hscBase))

// Per-layer positions (RecHitTools::getPositionLayer), indexed by the
// absolute layer number with offset. Row 0 is the barrel layer 0 entry.
//   z        endcap layer z at positive zside (0 beyond the endcap layers)
//   noseZ    HFNose layer z (0 when there is no nose geometry)
//   barrelX, barrelY  position of the first valid barrel cell of the layer
GENERATE_SOA_LAYOUT(TICLGeomLayersSoALayout,
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(float, noseZ),
                    SOA_COLUMN(float, barrelX),
                    SOA_COLUMN(float, barrelY))

using TICLGeomCommonSoA = TICLGeomCommonSoALayout<>;
using TICLGeomCommonSoAConstView = TICLGeomCommonSoA::ConstView;
using TICLGeomSiliconSoA = TICLGeomSiliconSoALayout<>;
using TICLGeomSiliconSoAConstView = TICLGeomSiliconSoA::ConstView;
using TICLGeomScintSoA = TICLGeomScintSoALayout<>;
using TICLGeomScintSoAConstView = TICLGeomScintSoA::ConstView;
using TICLGeomLookupSoA = TICLGeomLookupSoALayout<>;
using TICLGeomLookupSoAView = TICLGeomLookupSoA::View;
using TICLGeomLookupSoAConstView = TICLGeomLookupSoA::ConstView;
using TICLGeomLayersSoA = TICLGeomLayersSoALayout<>;
using TICLGeomLayersSoAConstView = TICLGeomLayersSoA::ConstView;

namespace ticlgeom {

  // Sentinel for wafer and cell coordinates of cells that have none
  // (RecHitTools returns int max there; the SoA stores int16).
  constexpr int16_t kInvalidCoord = std::numeric_limits<int16_t>::max();

  // Sensor group of a cell outside HGCal and HFNose (hgcal::UNKNOWN).
  constexpr int32_t kUnknownSensorGroup = 4;

  // Detid field layout and HGCal detector values (DetId::HGCalEE, HGCalHSi,
  // HGCalHSc), for device code that cannot use the DetId headers.
  namespace detail {
    constexpr uint32_t kDetOffset = 28, kDetMask = 0xF;
    constexpr uint32_t kHGCalEE = 8, kHGCalHSi = 9, kHGCalHSc = 10;
    // silicon: cellU/cellV occupy bits 0..9, so the wafer key is bits 10..27
    constexpr uint32_t kSiCellBits = 10, kSiKeyBits = kDetOffset - kSiCellBits;
    // scintillator: iphi occupies bits 0..8, so the ring key is bits 9..27
    constexpr uint32_t kScCellBits = 9, kScKeyBits = kDetOffset - kScCellBits;
    constexpr uint32_t kSiKeyMask = (1u << kSiKeyBits) - 1, kScKeyMask = (1u << kScKeyBits) - 1;
    constexpr int32_t kSiSlots = 1 << kSiKeyBits;  // 262144
    constexpr int32_t kScSlots = 1 << kScKeyBits;  // 524288
    constexpr int32_t kLookupSlots = 2 * kSiSlots + kScSlots;

    // Row of rawDetId in rows [lo, hi] of the common block, or -1 if absent.
    SOA_HOST_DEVICE SOA_INLINE int32_t searchRows(TICLGeomCommonSoAConstView const& common,
                                                  int32_t lo,
                                                  int32_t hi,
                                                  uint32_t rawDetId) {
      while (lo <= hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        const uint32_t val = common[mid].rawDetId();
        if (val == rawDetId) {
          return mid;
        }
        if (val < rawDetId) {
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      return -1;
    }
  }  // namespace detail

  // Row of rawDetId in the common block by binary search, or -1 if absent.
  SOA_HOST_DEVICE SOA_INLINE int32_t indexOf(TICLGeomCommonSoAConstView const& common, uint32_t rawDetId) {
    return detail::searchRows(common, 0, common.metadata().size() - 1, rawDetId);
  }

  // Row in the lookup block for an HGCal detid, or -1 for any other detid
  // or for an empty lookup block.
  SOA_HOST_DEVICE SOA_INLINE int32_t coarseSlot(TICLGeomLookupSoAConstView const& lookup, uint32_t rawDetId) {
    using namespace detail;
    if (lookup.metadata().size() == 0) {
      return -1;
    }
    const uint32_t det = (rawDetId >> kDetOffset) & kDetMask;
    if (det == kHGCalEE) {
      return lookup.eeBase() + static_cast<int32_t>((rawDetId >> kSiCellBits) & kSiKeyMask);
    } else if (det == kHGCalHSi) {
      return lookup.hsiBase() + static_cast<int32_t>((rawDetId >> kSiCellBits) & kSiKeyMask);
    } else if (det == kHGCalHSc) {
      return lookup.hscBase() + static_cast<int32_t>((rawDetId >> kScCellBits) & kScKeyMask);
    }
    return -1;
  }

  // Row of rawDetId in the common block, or -1 if absent. HGCal detids use
  // the lookup block; other detids use a binary search over all cells.
  SOA_HOST_DEVICE SOA_INLINE int32_t denseIdOf(TICLGeomLookupSoAConstView const& lookup,
                                               TICLGeomCommonSoAConstView const& common,
                                               uint32_t rawDetId) {
    const int32_t slot = coarseSlot(lookup, rawDetId);
    if (slot < 0) {
      return indexOf(common, rawDetId);
    }
    const int32_t start = lookup[slot].blockStart();
    return detail::searchRows(common, start, start + lookup[slot].blockCount() - 1, rawDetId);
  }

  struct Position {
    float x = 0.f, y = 0.f, z = 0.f;
    bool operator==(Position const&) const = default;
  };

  struct CellCoord {
    int32_t u = kInvalidCoord, v = kInvalidCoord;
    bool operator==(CellCoord const&) const = default;
  };

  struct ScintDEtaDPhi {
    float dEta = 0.f, dPhi = 0.f;
    bool operator==(ScintDEtaDPhi const&) const = default;
  };

  struct WaferInfo {
    int32_t type = 0, partialType = 0, orientation = 0, placementIndex = 0, cassette = 0;
    bool operator==(WaferInfo const&) const = default;
  };

  struct TileInfo {
    int32_t type = 0, sipm = 0, cassette = 0;
    bool operator==(TileInfo const&) const = default;
  };

  namespace detail {
    // Row of |layer| in the layers block, or -1 if |layer| is outside it.
    SOA_HOST_DEVICE SOA_INLINE int32_t layerRow(TICLGeomLayersSoAConstView const& layers, int32_t layer) {
      const int32_t n = layers.metadata().size();
      return (layer > -n && layer < n) ? ((layer < 0) ? -layer : layer) : -1;
    }

    // Endcap or HFNose layer z; a negative layer gives the negative side, a
    // layer outside the layers block gives 0.
    SOA_HOST_DEVICE SOA_INLINE float signedLayerZ(TICLGeomLayersSoAConstView const& layers, int32_t layer, bool nose) {
      const int32_t row = layerRow(layers, layer);
      if (row < 0) {
        return 0.f;
      }
      const float z = nose ? layers[row].noseZ() : layers[row].z();
      return (layer < 0) ? -z : z;
    }
  }  // namespace detail

  // One cell of the TICLGeom collection; use the alias ticlgeom::Cell. Get it
  // from the cell or cellAt method of the collection view; it is valid only
  // while that view exists. Do not return a Cell from a function that takes
  // the view by value.
  // A getter of an invalid cell returns a default value: 0, -1, false,
  // kInvalidCoord, kUnknownSensorGroup, or float max for radiusToSide. A
  // silicon or scintillator getter returns the same default for a cell of
  // the other type.
  template <typename GeomView>
  class CellOf {
  public:
    CellOf() = default;

    SOA_HOST_DEVICE SOA_INLINE bool valid() const { return row_ >= 0; }
    // Row of the cell in the collection (dense id), or -1.
    SOA_HOST_DEVICE SOA_INLINE int32_t index() const { return row_; }
    SOA_HOST_DEVICE SOA_INLINE uint32_t rawDetId() const { return valid() ? common().rawDetId() : 0u; }

    SOA_HOST_DEVICE SOA_INLINE Position position() const {
      if (!valid()) {
        return Position{};
      }
      auto const c = common();
      return Position{.x = c.x(), .y = c.y(), .z = c.z()};
    }
    SOA_HOST_DEVICE SOA_INLINE float eta(float vertexZ = 0.f) const {
      if (!valid()) {
        return 0.f;
      }
      const Position p = position();
      return xtd::asinh((p.z - vertexZ) / xtd::sqrt(p.x * p.x + p.y * p.y));
    }
    SOA_HOST_DEVICE SOA_INLINE float phi() const {
      if (!valid()) {
        return 0.f;
      }
      const Position p = position();
      return xtd::atan2(p.y, p.x);
    }
    SOA_HOST_DEVICE SOA_INLINE float pt(float energy, float vertexZ = 0.f) const {
      return valid() ? energy / xtd::cosh(eta(vertexZ)) : 0.f;
    }
    SOA_HOST_DEVICE SOA_INLINE int32_t zside() const { return valid() ? common().zside() : 0; }
    SOA_HOST_DEVICE SOA_INLINE int32_t layer() const { return valid() ? common().layer() : -1; }
    SOA_HOST_DEVICE SOA_INLINE int32_t layerWithOffset() const { return valid() ? common().layerWithOffset() : -1; }
    SOA_HOST_DEVICE SOA_INLINE int32_t type() const { return valid() ? common().cellType() : -1; }
    SOA_HOST_DEVICE SOA_INLINE int32_t sensorGroup() const {
      return valid() ? common().sensorGroup() : kUnknownSensorGroup;
    }
    SOA_HOST_DEVICE SOA_INLINE int32_t cassette() const { return valid() ? common().cassette() : 0; }
    SOA_HOST_DEVICE SOA_INLINE bool masked() const { return valid() && common().masked(); }
    SOA_HOST_DEVICE SOA_INLINE bool isSilicon() const { return valid() && common().isSilicon(); }
    SOA_HOST_DEVICE SOA_INLINE bool isScintillator() const { return valid() && common().isScintillator(); }
    SOA_HOST_DEVICE SOA_INLINE bool isBarrel() const { return valid() && common().isBarrel(); }

    // silicon cells
    SOA_HOST_DEVICE SOA_INLINE float thickness() const {
      const int32_t i = siliconRow();
      return (i >= 0) ? geom_->silicon()[i].siThickness() : 0.f;
    }
    SOA_HOST_DEVICE SOA_INLINE int32_t thicknessIndex() const {
      const int32_t i = siliconRow();
      return (i >= 0) ? geom_->silicon()[i].siThickIndex() : -1;
    }
    SOA_HOST_DEVICE SOA_INLINE float radiusToSide() const {
      const int32_t i = siliconRow();
      return (i >= 0) ? geom_->silicon()[i].radiusToSide() : std::numeric_limits<float>::max();
    }
    SOA_HOST_DEVICE SOA_INLINE CellCoord wafer() const {
      const int32_t i = siliconRow();
      if (i < 0) {
        return CellCoord{};
      }
      auto const si = geom_->silicon()[i];
      return CellCoord{.u = si.waferU(), .v = si.waferV()};
    }
    SOA_HOST_DEVICE SOA_INLINE CellCoord uv() const {
      const int32_t i = siliconRow();
      if (i < 0) {
        return CellCoord{};
      }
      auto const si = geom_->silicon()[i];
      return CellCoord{.u = si.cellU(), .v = si.cellV()};
    }
    SOA_HOST_DEVICE SOA_INLINE bool isHalfCell() const {
      const int32_t i = siliconRow();
      return (i >= 0) && geom_->silicon()[i].isHalfCell();
    }
    SOA_HOST_DEVICE SOA_INLINE WaferInfo waferInfo() const {
      const int32_t i = siliconRow();
      if (i < 0) {
        return WaferInfo{};
      }
      auto const si = geom_->silicon()[i];
      return WaferInfo{.type = si.waferType(),
                       .partialType = si.waferPartialType(),
                       .orientation = si.waferOrientation(),
                       .placementIndex = si.waferPlacementIndex(),
                       .cassette = cassette()};
    }

    // scintillator cells
    SOA_HOST_DEVICE SOA_INLINE ScintDEtaDPhi dEtaDPhi() const {
      const int32_t i = scintRow();
      if (i < 0) {
        return ScintDEtaDPhi{};
      }
      auto const sc = geom_->scint()[i];
      return ScintDEtaDPhi{.dEta = sc.scintDEta(), .dPhi = sc.scintDPhi()};
    }
    SOA_HOST_DEVICE SOA_INLINE int32_t maxIphi() const {
      const int32_t i = scintRow();
      return (i >= 0) ? geom_->scint()[i].scintMaxIphi() : 0;
    }
    SOA_HOST_DEVICE SOA_INLINE bool isFine() const {
      const int32_t i = scintRow();
      return (i >= 0) && geom_->scint()[i].isScintillatorFine();
    }
    SOA_HOST_DEVICE SOA_INLINE TileInfo tileInfo() const {
      const int32_t i = scintRow();
      if (i < 0) {
        return TileInfo{};
      }
      auto const sc = geom_->scint()[i];
      return TileInfo{.type = sc.tileType(), .sipm = sc.tileSipm(), .cassette = cassette()};
    }

  private:
    friend GeomView;
    SOA_HOST_DEVICE SOA_INLINE CellOf(GeomView const* geom, int32_t row) : geom_(geom), row_(row) {}

    // Common-block element of a valid cell.
    SOA_HOST_DEVICE SOA_INLINE auto common() const { return geom_->common()[row_]; }
    SOA_HOST_DEVICE SOA_INLINE int32_t siliconRow() const {
      return isSilicon() ? row_ - geom_->common().nBarrel() : -1;
    }
    SOA_HOST_DEVICE SOA_INLINE int32_t scintRow() const {
      return isScintillator() ? row_ - geom_->common().nBarrel() - geom_->common().nSilicon() : -1;
    }

    GeomView const* geom_ = nullptr;
    int32_t row_ = -1;
  };

}  // namespace ticlgeom

// The TICLGeom collection. Its ConstView is the interface for host and
// device code; pass it by value to a kernel.
// clang-format off
GENERATE_SOA_BLOCKS(TICLGeomSoALayout,
                    SOA_BLOCK(common, TICLGeomCommonSoALayout),
                    SOA_BLOCK(silicon, TICLGeomSiliconSoALayout),
                    SOA_BLOCK(scint, TICLGeomScintSoALayout),
                    SOA_BLOCK(lookup, TICLGeomLookupSoALayout),
                    SOA_BLOCK(layers, TICLGeomLayersSoALayout),
                    SOA_CONST_VIEW_METHODS(
                        // The cell of rawDetId; an invalid cell if the collection does not hold it.
                        // A cell points to its view, so a temporary view cannot give one.
                        SOA_HOST_DEVICE SOA_INLINE auto cell(uint32_t rawDetId) const& {
                          return ticlgeom::CellOf<std::remove_cvref_t<decltype(*this)>>(
                              this, ticlgeom::denseIdOf(this->lookup(), this->common(), rawDetId));
                        }
                        void cell(uint32_t) const&& = delete;
                        // The cell at row index; an invalid cell if index is outside [0, size()).
                        SOA_HOST_DEVICE SOA_INLINE auto cellAt(int32_t index) const& {
                          return ticlgeom::CellOf<std::remove_cvref_t<decltype(*this)>>(
                              this, (index >= 0 && index < size()) ? index : -1);
                        }
                        void cellAt(int32_t) const&& = delete;
                        SOA_HOST_DEVICE SOA_INLINE int32_t size() const { return this->common().metadata().size(); }

                        // Endcap (layerZ) or HFNose (noseLayerZ) layer z; a negative layer gives
                        // the z of the negative side. 0 for a layer outside the endcap.
                        SOA_HOST_DEVICE SOA_INLINE float layerZ(int32_t layer) const {
                          return ticlgeom::detail::signedLayerZ(this->layers(), layer, false);
                        }
                        SOA_HOST_DEVICE SOA_INLINE float noseLayerZ(int32_t layer) const {
                          return ticlgeom::detail::signedLayerZ(this->layers(), layer, true);
                        }
                        // Position of the first valid barrel cell of layer |layer|; 0 for a
                        // layer outside the barrel.
                        SOA_HOST_DEVICE SOA_INLINE ticlgeom::Position barrelLayerPosition(int32_t layer) const {
                          auto const layers = this->layers();
                          const int32_t row = ticlgeom::detail::layerRow(layers, layer);
                          return (row >= 0)
                                     ? ticlgeom::Position{.x = layers[row].barrelX(), .y = layers[row].barrelY()}
                                     : ticlgeom::Position{};
                        }

                        SOA_HOST_DEVICE SOA_INLINE int32_t numberOfLayers() const { return this->common().numberOfLayers(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t lastLayerEE() const { return this->common().lastLayerEE(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t lastLayerFH() const { return this->common().lastLayerFH(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t firstLayerBH() const { return this->common().firstLayerBH(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t lastLayerBH() const { return this->common().lastLayerBH(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t lastLayerECAL() const { return this->common().lastLayerECAL(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t lastLayerBarrel() const { return this->common().lastLayerBarrel(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t noseLastLayer() const { return this->common().noseLastLayer(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t maxWafersPerLayer() const {
                          return this->common().maxNumberOfWafersPerLayer();
                        }
                        SOA_HOST_DEVICE SOA_INLINE int32_t noseMaxWafersPerLayer() const {
                          return this->common().maxNumberOfWafersNose();
                        }
                        SOA_HOST_DEVICE SOA_INLINE int32_t scintMaxIphi() const { return this->common().bhMaxIphi(); }
                        SOA_HOST_DEVICE SOA_INLINE int32_t geometryType() const { return this->common().geometryType(); }))
// clang-format on

using TICLGeomSoA = TICLGeomSoALayout<>;
using TICLGeomSoAView = TICLGeomSoA::View;
using TICLGeomSoAConstView = TICLGeomSoA::ConstView;

namespace ticlgeom {
  using Cell = CellOf<TICLGeomSoAConstView>;
}  // namespace ticlgeom

#endif  // CondFormats_HGCalObjects_interface_TICLGeomSoA_h
