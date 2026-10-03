#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
// also brings in the CopyToDevice specialization that registers the
// automatic host to device transfer of the produced collection
#include "CondFormats/HGCalObjects/interface/alpaka/TICLGeomDevice.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "oneapi/tbb.h"
#include "oneapi/tbb/task_arena.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Flattens every RecHitTools per-cell quantity of the selected
  // calorimeters into the TICLGeom SoABlocks (common | silicon | scint),
  // ordered by increasing rawDetId so cells group as [barrel|silicon|scint]
  // and each cell-type-specific column is allocated only for its cells.
  // Columns are filled by calling RecHitTools, so the replacement is exact
  // by construction; methods that would crash or log errors outside their
  // detector family are guarded and their cells carry the documented
  // sentinel values.
  class TICLGeomESProducer : public ESProducer {
  public:
    explicit TICLGeomESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig), detectors_(iConfig.getParameter<std::vector<std::string>>("detectors")) {
      auto cc = setWhatProduced(this);
      geomToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<std::string>>("detectors", {"HGCal"})
          ->setComment(
              "Detectors or subdetectors to include (valid options: ECAL, HCAL, HGCal, HFNose, EB, EE, ES, HB, HE, "
              "HF, HO, HGCEE, HGCHESil, HGCHESci)");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<TICLGeomHost> produce(CaloGeometryRecord const& iRecord) {
      auto const& geom = iRecord.get(geomToken_);
      hgcal::RecHitTools tools;
      tools.setGeometry(geom);

      // Map of detector names to pair of DetId::Detector and subdet id
      static const std::map<std::string, std::pair<DetId::Detector, int>> detMap = {{"EB", {DetId::Ecal, 1}},
                                                                                    {"EE", {DetId::Ecal, 2}},
                                                                                    {"ES", {DetId::Ecal, 3}},
                                                                                    {"HB", {DetId::Hcal, 1}},
                                                                                    {"HE", {DetId::Hcal, 2}},
                                                                                    {"HF", {DetId::Hcal, 4}},
                                                                                    {"HO", {DetId::Hcal, 3}},
                                                                                    {"HGCEE", {DetId::HGCalEE, 0}},
                                                                                    {"HGCHESil", {DetId::HGCalHSi, 0}},
                                                                                    {"HGCHESci", {DetId::HGCalHSc, 0}},
                                                                                    {"HFNose", {DetId::Forward, 6}}};

      static const std::map<std::string, std::vector<std::string>> detGroups = {
          {"ECAL", {"EB", "EE", "ES"}},
          {"HCAL", {"HB", "HE", "HF", "HO"}},
          {"HGCal", {"HGCEE", "HGCHESil", "HGCHESci"}},
          {"HFNose", {"HFNose"}}};

      std::vector<DetId> validIds;
      auto appendIds = [&](std::string const& det) {
        auto const it = detMap.find(det);
        if (it == detMap.end()) {
          throw cms::Exception("TICLGeomInvalidDetector") << "Detector " << det << " is not a valid detector name";
        }
        const auto& ids = geom.getValidDetIds(it->second.first, it->second.second);
        validIds.insert(validIds.end(), ids.begin(), ids.end());
      };

      for (const auto& group : detectors_) {
        auto const it = detGroups.find(group);
        if (it != detGroups.end()) {
          for (const auto& det : it->second) {
            appendIds(det);
          }
        } else {
          appendIds(group);
        }
      }

      std::sort(validIds.begin(), validIds.end(), [](DetId a, DetId b) { return a.rawId() < b.rawId(); });

      // count cells per block; rawDetId order groups them [barrel|silicon|scint]
      int32_t nSilicon = 0, nScint = 0;
      for (const auto id : validIds) {
        if (tools.isSilicon(id)) {
          ++nSilicon;
        } else if (tools.isScintillator(id)) {
          ++nScint;
        }
      }
      const int32_t nCells = static_cast<int32_t>(validIds.size());
      const int32_t nBarrel = nCells - nSilicon - nScint;

      auto product =
          std::make_unique<TICLGeomHost>(cms::alpakatools::host(), std::array<int32_t, 3>{nCells, nSilicon, nScint});
      auto view = product->view();
      auto common = view.common();
      auto silicon = view.silicon();
      auto scint = view.scint();

      // Fill the per-cell columns in parallel. Each i writes disjoint rows:
      // common[i], and for a silicon/scint cell the block-local row whose index
      // is fixed by the [barrel|silicon|scint] rawDetId ordering (i - nBarrel,
      // i - nBarrel - nSilicon). RecHitTools const access is thread-safe.
      tbb::this_task_arena::isolate([&] {
        tbb::parallel_for(int32_t(0), nCells, [&](int32_t i) {
          const DetId id = validIds[i];
          auto cell = common[i];

          const bool isHGCal =
              (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi || id.det() == DetId::HGCalHSc);
          const bool isNose = (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose);
          const bool isSi = tools.isSilicon(id);
          const bool isSc = tools.isScintillator(id);

          const auto pos = tools.getPosition(id);
          cell.rawDetId() = id.rawId();
          cell.x() = pos.x();
          cell.y() = pos.y();
          cell.z() = pos.z();
          cell.zside() = tools.zside(id);
          cell.layer() = tools.getLayer(id);
          cell.layerWithOffset() = tools.getLayerWithOffset(id);
          cell.isSilicon() = isSi;
          cell.isScintillator() = isSc;
          cell.isBarrel() = tools.isBarrel(id);

          const auto waferInfo = tools.getWaferInfo(id);
          const auto tileInfo = tools.getTileInfo(id);
          cell.cassette() = isSc ? tileInfo.cassette : waferInfo.cassette;

          // getCellType and getSensorGroup are only defined for HGCal and
          // HFNose; maskCell casts the subdetector geometry to HGCalGeometry
          if (isHGCal || isNose) {
            cell.cellType() = tools.getCellType(id);
            cell.sensorGroup() = tools.getSensorGroup(id);
            cell.masked() = tools.maskCell(id);
          } else {
            cell.cellType() = -1;
            cell.sensorGroup() = hgcal::UNKNOWN;
            cell.masked() = false;
          }

          // silicon-only columns; the block-local index is i - nBarrel, valid
          // because the rawDetId order is [barrel|silicon|scint]
          if (isSi) {
            assert(i >= nBarrel && (i - nBarrel) < nSilicon);
            auto si = silicon[i - nBarrel];
            si.siThickness() = tools.getSiThickness(id);
            si.siThickIndex() = tools.getSiThickIndex(id);
            si.radiusToSide() = tools.getRadiusToSide(id);
            const auto wafer = tools.getWafer(id);
            const auto cellUV = tools.getCell(id);
            si.waferU() = wafer.first;
            si.waferV() = wafer.second;
            si.cellU() = cellUV.first;
            si.cellV() = cellUV.second;
            si.waferType() = waferInfo.type;
            si.waferPartialType() = waferInfo.partialType;
            si.waferOrientation() = waferInfo.orientation;
            si.waferPlacementIndex() = waferInfo.placementIndex;
            si.isHalfCell() = tools.isHalfCell(id);
          } else if (isSc) {
            assert(i >= (nBarrel + nSilicon));
            auto sc = scint[i - nBarrel - nSilicon];
            const auto dEtaDPhi = tools.getScintDEtaDPhi(id);
            sc.scintDEta() = dEtaDPhi.first;
            sc.scintDPhi() = dEtaDPhi.second;
            sc.scintMaxIphi() = tools.getScintMaxIphi(id);
            sc.tileType() = tileInfo.type;
            sc.tileSipm() = tileInfo.sipm;
            sc.isScintillatorFine() = tools.isScintillatorFine(id);
          }
        });
      });

      common.lastLayerEE() = tools.lastLayerEE();
      common.lastLayerFH() = tools.lastLayerFH();
      common.firstLayerBH() = tools.firstLayerBH();
      common.lastLayerBH() = tools.lastLayerBH();
      common.numberOfLayers() = tools.getNumberOfLayers();
      common.lastLayerECAL() = tools.lastLayerECAL();
      common.lastLayerBarrel() = tools.lastLayerBarrel();
      common.maxNumberOfWafersPerLayer() = tools.maxNumberOfWafersPerLayer();
      common.bhMaxIphi() = tools.getScintMaxIphi();
      common.geometryType() = tools.getGeometryType();
      common.noseLastLayer() = tools.lastLayer(true);
      common.maxNumberOfWafersNose() = tools.maxNumberOfWafersPerLayer(true);
      common.nBarrel() = nBarrel;
      common.nSilicon() = nSilicon;

      return product;
    }

  private:
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
    std::vector<std::string> detectors_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TICLGeomESProducer);
