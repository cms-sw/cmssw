#include <limits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TICLGeomDeviceCheck.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace ticlgeomtest {

    namespace {

      // Cell fields: silicon cellU (bits 0-4) and scintillator iphi (bits 0-8).
      constexpr uint32_t kSiCellUMask = 0x1F;
      constexpr uint32_t kScPhiMask = 0x1FF;

      // Detids that no HGCal collection holds: null, invalid detector, and
      // EE, HSc and EB ids with no cell.

      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t missingIdAt(int k) {
        constexpr uint32_t kEcalBarrel = (3u << ticlgeom::detail::kDetOffset) | (1u << 25);
        switch (k) {
          case 0:
            return 0u;
          case 1:
            return 0xFFFFFFFFu;
          case 2:
            return ticlgeom::detail::kHGCalEE << ticlgeom::detail::kDetOffset;
          case 3:
            return ticlgeom::detail::kHGCalHSc << ticlgeom::detail::kDetOffset;
          default:
            return kEcalBarrel;
        }
      }

      // True if every getter of an invalid cell returns its default value.
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isDefault(ticlgeom::Cell const& c) {
        return !c.valid() && c.index() == -1 && c.rawDetId() == 0u && c.position() == ticlgeom::Position{} &&
               c.eta() == 0.f && c.phi() == 0.f && c.pt(kHitEnergy) == 0.f && c.zside() == 0 && c.layer() == -1 &&
               c.layerWithOffset() == -1 && c.type() == -1 && c.sensorGroup() == ticlgeom::kUnknownSensorGroup &&
               c.cassette() == 0 && !c.masked() && !c.isSilicon() && !c.isScintillator() && !c.isBarrel() &&
               c.thickness() == 0.f && c.thicknessIndex() == -1 &&
               c.radiusToSide() == std::numeric_limits<float>::max() && c.wafer() == ticlgeom::CellCoord{} &&
               c.uv() == ticlgeom::CellCoord{} && !c.isHalfCell() && c.waferInfo() == ticlgeom::WaferInfo{} &&
               c.dEtaDPhi() == ticlgeom::ScintDEtaDPhi{} && c.maxIphi() == 0 && !c.isFine() &&
               c.tileInfo() == ticlgeom::TileInfo{};
      }

    }  // namespace

    class CheckLookupKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    TICLGeomCommonSoAConstView common,
                                    TICLGeomLookupSoAConstView lookup,
                                    int32_t* nBad) const {
        for (auto i : cms::alpakatools::uniform_elements(acc, common.metadata().size())) {
          const uint32_t rawDetId = common[i].rawDetId();
          if (ticlgeom::indexOf(common, rawDetId) != static_cast<int32_t>(i) ||
              ticlgeom::denseIdOf(lookup, common, rawDetId) != static_cast<int32_t>(i)) {
            alpaka::atomicAdd(acc, nBad, 1, alpaka::hierarchy::Blocks{});
          }
        }
      }
    };

    class CellsKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, TICLGeomSoAConstView geom, CellResult* out) const {
        for (auto i : cms::alpakatools::uniform_elements(acc, geom.size())) {
          const ticlgeom::Cell at = geom.cellAt(i);
          const ticlgeom::Cell c = geom.cell(at.rawDetId());
          auto& r = out[i];
          r.index = c.index();
          r.indexAt = at.index();
          r.rawDetId = c.rawDetId();
          r.position = c.position();
          r.eta = c.eta();
          r.etaDisplaced = c.eta(kVertexZ);
          r.phi = c.phi();
          r.pt = c.pt(kHitEnergy);
          r.ptDisplaced = c.pt(kHitEnergy, kVertexZ);
          r.zside = c.zside();
          r.layer = c.layer();
          r.layerWithOffset = c.layerWithOffset();
          r.type = c.type();
          r.sensorGroup = c.sensorGroup();
          r.cassette = c.cassette();
          r.masked = c.masked();
          r.isSilicon = c.isSilicon();
          r.isScintillator = c.isScintillator();
          r.isBarrel = c.isBarrel();
          r.thickness = c.thickness();
          r.thicknessIndex = c.thicknessIndex();
          r.radiusToSide = c.radiusToSide();
          r.wafer = c.wafer();
          r.uv = c.uv();
          r.isHalfCell = c.isHalfCell();
          r.waferInfo = c.waferInfo();
          r.dEtaDPhi = c.dEtaDPhi();
          r.maxIphi = c.maxIphi();
          r.isFine = c.isFine();
          r.tileInfo = c.tileInfo();
        }
      }
    };

    class LayersKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    TICLGeomSoAConstView geom,
                                    int32_t nLayers,
                                    LayerResult* out) const {
        for (auto l : cms::alpakatools::uniform_elements(acc, nLayers)) {
          const int32_t layer = static_cast<int32_t>(l);
          out[l].zPlus = geom.layerZ(layer);
          out[l].zMinus = geom.layerZ(-layer);
          out[l].noseZ = geom.noseLayerZ(layer);
          out[l].barrel = geom.barrelLayerPosition(layer);
        }
      }
    };

    class ScalarsKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, TICLGeomSoAConstView geom, ScalarResult* out) const {
        for ([[maybe_unused]] auto unused : cms::alpakatools::uniform_elements(acc, 1)) {
          auto& r = *out;
          r.size = geom.size();
          r.numberOfLayers = geom.numberOfLayers();
          r.lastLayerEE = geom.lastLayerEE();
          r.lastLayerFH = geom.lastLayerFH();
          r.firstLayerBH = geom.firstLayerBH();
          r.lastLayerBH = geom.lastLayerBH();
          r.lastLayerECAL = geom.lastLayerECAL();
          r.lastLayerBarrel = geom.lastLayerBarrel();
          r.noseLastLayer = geom.noseLastLayer();
          r.maxWafersPerLayer = geom.maxWafersPerLayer();
          r.noseMaxWafersPerLayer = geom.noseMaxWafersPerLayer();
          r.scintMaxIphi = geom.scintMaxIphi();
          r.geometryType = geom.geometryType();
          r.nBadInvalid = isDefault(ticlgeom::Cell{}) ? 0 : 1;
          for (int k = 0; k < kNMissingIds; ++k) {
            if (!isDefault(geom.cell(missingIdAt(k)))) {
              ++r.nBadInvalid;
            }
          }
          if (!isDefault(geom.cellAt(-1)) || !isDefault(geom.cellAt(geom.size()))) {
            ++r.nBadInvalid;
          }
          // An absent cell in a filled wafer (cellU 31) and in a filled ring (iphi 511).
          bool siliconDone = false, scintDone = false;
          for (int32_t i = 0; i < geom.size() && !(siliconDone && scintDone); ++i) {
            const uint32_t raw = geom.cellAt(i).rawDetId();
            const uint32_t det = raw >> ticlgeom::detail::kDetOffset;
            const bool silicon = det == ticlgeom::detail::kHGCalEE || det == ticlgeom::detail::kHGCalHSi;
            const bool scint = det == ticlgeom::detail::kHGCalHSc;
            if ((silicon && !siliconDone) || (scint && !scintDone)) {
              const uint32_t cellMask = silicon ? kSiCellUMask : kScPhiMask;
              if (!isDefault(geom.cell(raw | cellMask))) {
                ++r.nBadInvalid;
              }
              siliconDone = siliconDone || silicon;
              scintDone = scintDone || scint;
            }
          }
          constexpr int32_t kMinLayer = std::numeric_limits<int32_t>::min();
          if (geom.layerZ(kMinLayer) != 0.f || geom.noseLayerZ(kMinLayer) != 0.f ||
              !(geom.barrelLayerPosition(kMinLayer) == ticlgeom::Position{})) {
            ++r.nBadInvalid;
          }
        }
      }
    };

    int32_t checkLookup(Queue& queue, TICLGeomCommonSoAConstView common, TICLGeomLookupSoAConstView lookup) {
      auto nBad = cms::alpakatools::make_device_buffer<int32_t>(queue);
      alpaka::memset(queue, nBad, 0);

      const auto workDiv =
          cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(common.metadata().size(), 256), 256);
      alpaka::exec<Acc1D>(queue, workDiv, CheckLookupKernel{}, common, lookup, nBad.data());

      auto nBadHost = cms::alpakatools::make_host_buffer<int32_t>(queue);
      alpaka::memcpy(queue, nBadHost, nBad);
      alpaka::wait(queue);
      return *nBadHost;
    }

    GeomResults runGeomQueries(Queue& queue, TICLGeomSoAConstView geom) {
      const int32_t nCells = geom.size();

      auto cells = cms::alpakatools::make_device_buffer<CellResult[]>(queue, nCells);
      const int32_t nLayers = geom.layers().metadata().size() + 1;
      auto layers = cms::alpakatools::make_device_buffer<LayerResult[]>(queue, nLayers);
      auto scalars = cms::alpakatools::make_device_buffer<ScalarResult>(queue);

      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nCells, 256), 256),
                          CellsKernel{},
                          geom,
                          cells.data());
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nLayers, 64), 64),
                          LayersKernel{},
                          geom,
                          nLayers,
                          layers.data());
      alpaka::exec<Acc1D>(queue, cms::alpakatools::make_workdiv<Acc1D>(1, 1), ScalarsKernel{}, geom, scalars.data());

      GeomResults results{nLayers,
                          cms::alpakatools::make_host_buffer<CellResult[]>(queue, nCells),
                          cms::alpakatools::make_host_buffer<LayerResult[]>(queue, nLayers),
                          cms::alpakatools::make_host_buffer<ScalarResult>(queue)};
      alpaka::memcpy(queue, results.cells, cells);
      alpaka::memcpy(queue, results.layers, layers);
      alpaka::memcpy(queue, results.scalars, scalars);
      alpaka::wait(queue);
      return results;
    }

  }  // namespace ticlgeomtest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
