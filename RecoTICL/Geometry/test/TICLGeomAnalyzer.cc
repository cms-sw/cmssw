#include <cmath>
#include <limits>
#include <string>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

// Closure test for the TICLGeom SoA: checks the rawDetId ordering, both
// lookups (binary search and dense id hash table), and every column and
// scalar against the RecHitTools method it replaces.
class TICLGeomAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TICLGeomAnalyzer(const edm::ParameterSet& iConfig)
      : label_(iConfig.getParameter<std::string>("label")),
        ticlGeomToken_(esConsumes(edm::ESInputTag("", label_))),
        ticlGeomLookupToken_(esConsumes(edm::ESInputTag("", label_))),
        ticlGeomLayersToken_(esConsumes(edm::ESInputTag("", ""))),
        caloGeomToken_(esConsumes()) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "");
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(const edm::Event&, const edm::EventSetup& iSetup) override {
    auto const& ticlGeom = iSetup.getData(ticlGeomToken_);
    auto const& caloGeom = iSetup.getData(caloGeomToken_);
    auto const& blocks = ticlGeom.const_view();
    auto const common = blocks.common();
    auto const silicon = blocks.silicon();
    auto const scint = blocks.scint();
    auto const& lookup = iSetup.getData(ticlGeomLookupToken_).const_view();

    hgcal::RecHitTools tools;
    tools.setGeometry(caloGeom);

    ticlgeom::Tools facade;
    facade.setGeometry(ticlGeom, iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));

    const int32_t n = common.metadata().size();
    constexpr float tol = 1.e-4f;

    for (int32_t i = 0; i < n; ++i) {
      const uint32_t rawId = common[i].rawDetId();
      const DetId id(rawId);

      auto require = [&](bool ok, const char* what) {
        if (!ok) {
          throw cms::Exception("TICLGeomClosure")
              << "label '" << label_ << "': column '" << what << "' does not match RecHitTools for detid " << rawId
              << " at dense id " << i;
        }
      };

      if (i > 0 && common[i - 1].rawDetId() >= rawId) {
        throw cms::Exception("TICLGeomOrdering")
            << "label '" << label_ << "': rawDetId not strictly increasing at index " << i;
      }
      require(ticlgeom::indexOf(common, rawId) == i, "indexOf");
      require(ticlgeom::denseIdOf(lookup, common, rawId) == i, "denseIdOf");

      const auto pos = tools.getPosition(id);
      require(std::abs(common[i].x() - pos.x()) < tol, "x");
      require(std::abs(common[i].y() - pos.y()) < tol, "y");
      require(std::abs(common[i].z() - pos.z()) < tol, "z");
      // the facade and SoA helpers derive eta and phi from x,y,z; check
      // they match RecHitTools
      require(std::abs(ticlgeom::etaFromVertex(common, i, 0.f) - tools.getEta(id)) < tol, "etaFromVertex");
      require(std::abs(facade.getPhi(id) - tools.getPhi(id)) < tol, "facade getPhi");
      require(std::abs(facade.getEta(id) - tools.getEta(id)) < tol, "facade getEta");
      require(std::abs(ticlgeom::pt(common, i, 1.f, 0.f) - tools.getPt(id, 1.f, 0.f)) < tol, "pt");

      require(common[i].zside() == tools.zside(id), "zside");
      require(common[i].layer() == static_cast<int16_t>(tools.getLayer(id)), "layer");
      require(common[i].layerWithOffset() == static_cast<int16_t>(tools.getLayerWithOffset(id)), "layerWithOffset");

      const bool isSi = tools.isSilicon(id);
      const bool isSc = tools.isScintillator(id);
      require(common[i].isSilicon() == isSi, "isSilicon");
      require(common[i].isScintillator() == isSc, "isScintillator");
      require(common[i].isBarrel() == tools.isBarrel(id), "isBarrel");
      require(common[i].cassette() == (isSc ? tools.getTileInfo(id).cassette : tools.getWaferInfo(id).cassette),
              "cassette");

      // silicon-only columns live in the silicon block at i - nBarrel
      if (isSi) {
        auto const si = silicon[i - common.nBarrel()];
        require(si.siThickness() == tools.getSiThickness(id), "siThickness");
        require(si.siThickIndex() == tools.getSiThickIndex(id), "siThickIndex");
        require(si.radiusToSide() == tools.getRadiusToSide(id), "radiusToSide");
        const auto wafer = tools.getWafer(id);
        const auto cellUV = tools.getCell(id);
        require(si.waferU() == wafer.first && si.waferV() == wafer.second, "waferUV");
        require(si.cellU() == cellUV.first && si.cellV() == cellUV.second, "cellUV");
        const auto waferInfo = tools.getWaferInfo(id);
        require(si.waferType() == waferInfo.type, "waferType");
        require(si.waferPartialType() == waferInfo.partialType, "waferPartialType");
        require(si.waferOrientation() == waferInfo.orientation, "waferOrientation");
        require(si.waferPlacementIndex() == waferInfo.placementIndex, "waferPlacementIndex");
        require(si.isHalfCell() == tools.isHalfCell(id), "isHalfCell");
      }

      // scintillator-only columns live in the scint block at i - nBarrel - nSilicon
      if (isSc) {
        auto const sc = scint[i - common.nBarrel() - common.nSilicon()];
        const auto dEtaDPhi = tools.getScintDEtaDPhi(id);
        require(sc.scintDEta() == dEtaDPhi.first, "scintDEta");
        require(sc.scintDPhi() == dEtaDPhi.second, "scintDPhi");
        require(sc.scintMaxIphi() == static_cast<int16_t>(tools.getScintMaxIphi(id)), "scintMaxIphi");
        const auto tileInfo = tools.getTileInfo(id);
        require(sc.tileType() == tileInfo.type, "tileType");
        require(sc.tileSipm() == tileInfo.sipm, "tileSipm");
        require(sc.isScintillatorFine() == tools.isScintillatorFine(id), "isScintillatorFine");
      }

      // the facade must reproduce RecHitTools through its block routing
      require(facade.denseId(id) == i, "facade denseId");
      require(std::abs(facade.getPosition(id).z() - pos.z()) < tol, "facade getPosition");
      require(facade.zside(id) == tools.zside(id), "facade zside");
      require(facade.getLayer(id) == tools.getLayer(id), "facade getLayer");
      require(facade.getLayerWithOffset(id) == tools.getLayerWithOffset(id), "facade getLayerWithOffset");
      require(facade.isSilicon(id) == isSi, "facade isSilicon");
      require(facade.isScintillator(id) == isSc, "facade isScintillator");
      require(facade.isBarrel(id) == tools.isBarrel(id), "facade isBarrel");
      require(facade.isHalfCell(id) == tools.isHalfCell(id), "facade isHalfCell");
      require(facade.isScintillatorFine(id) == tools.isScintillatorFine(id), "facade isScintillatorFine");
      require(facade.getScintDEtaDPhi(id) == tools.getScintDEtaDPhi(id), "facade getScintDEtaDPhi");
      require(facade.getScintMaxIphi(id) == tools.getScintMaxIphi(id), "facade getScintMaxIphi");
      if (isSi) {
        require(facade.getSiThickIndex(id) == tools.getSiThickIndex(id), "facade getSiThickIndex");
        require(facade.getWafer(id) == tools.getWafer(id), "facade getWafer");
        require(facade.getCell(id) == tools.getCell(id), "facade getCell");
        const auto a = facade.getWaferInfo(id);
        const auto b = tools.getWaferInfo(id);
        require(a.type == b.type && a.partialType == b.partialType && a.orientation == b.orientation &&
                    a.placementIndex == b.placementIndex && a.cassette == b.cassette,
                "facade getWaferInfo");
      }
      if (isSc) {
        const auto a = facade.getTileInfo(id);
        const auto b = tools.getTileInfo(id);
        require(a.type == b.type && a.sipm == b.sipm && a.cassette == b.cassette, "facade getTileInfo");
      }

      const bool isHGCal = (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi || id.det() == DetId::HGCalHSc);
      const bool isNose = (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose);
      if (isHGCal || isNose) {
        // getCellType / getSensorGroup / maskCell crash in RecHitTools on
        // barrel detids (they cast to HGCalGeometry), so only compare here
        require(common[i].cellType() == tools.getCellType(id), "cellType");
        require(common[i].sensorGroup() == tools.getSensorGroup(id), "sensorGroup");
        require(common[i].masked() == tools.maskCell(id), "masked");
        require(facade.getCellType(id) == tools.getCellType(id), "facade getCellType");
        require(facade.getSensorGroup(id) == tools.getSensorGroup(id), "facade getSensorGroup");
        require(facade.maskCell(id) == tools.maskCell(id), "facade maskCell");
        require(facade.getSiThickness(id) == tools.getSiThickness(id), "facade getSiThickness");
        require(facade.getRadiusToSide(id) == tools.getRadiusToSide(id), "facade getRadiusToSide");
      } else {
        require(common[i].cellType() == -1, "cellType sentinel");
        require(common[i].sensorGroup() == hgcal::UNKNOWN, "sensorGroup sentinel");
        require(common[i].masked() == false, "masked sentinel");
      }
    }

    if (ticlgeom::denseIdOf(lookup, common, 0u) != -1 || ticlgeom::denseIdOf(lookup, common, 0xFFFFFFFFu) != -1) {
      throw cms::Exception("TICLGeomLookup") << "label '" << label_ << "': denseIdOf did not miss on invalid detids";
    }

    auto requireScalar = [&](int32_t stored, int32_t expected, const char* what) {
      if (stored != expected) {
        throw cms::Exception("TICLGeomClosure")
            << "label '" << label_ << "': scalar '" << what << "' is " << stored << ", RecHitTools says " << expected;
      }
    };
    requireScalar(common.lastLayerEE(), tools.lastLayerEE(), "lastLayerEE");
    requireScalar(common.lastLayerFH(), tools.lastLayerFH(), "lastLayerFH");
    requireScalar(common.firstLayerBH(), tools.firstLayerBH(), "firstLayerBH");
    requireScalar(common.lastLayerBH(), tools.lastLayerBH(), "lastLayerBH");
    requireScalar(common.numberOfLayers(), tools.getNumberOfLayers(), "numberOfLayers");
    requireScalar(common.lastLayerECAL(), tools.lastLayerECAL(), "lastLayerECAL");
    requireScalar(common.lastLayerBarrel(), tools.lastLayerBarrel(), "lastLayerBarrel");
    requireScalar(common.maxNumberOfWafersPerLayer(), tools.maxNumberOfWafersPerLayer(), "maxNumberOfWafersPerLayer");
    requireScalar(common.bhMaxIphi(), tools.getScintMaxIphi(), "bhMaxIphi");
    requireScalar(common.geometryType(), tools.getGeometryType(), "geometryType");
    requireScalar(facade.lastLayerEE(), tools.lastLayerEE(), "facade lastLayerEE");
    requireScalar(facade.lastLayer(), tools.lastLayer(), "facade lastLayer");
    requireScalar(facade.lastLayer(true), tools.lastLayer(true), "facade lastLayer nose");
    requireScalar(
        facade.maxNumberOfWafersPerLayer(true), tools.maxNumberOfWafersPerLayer(true), "facade maxNumberOfWafersNose");

    for (int lay = 1; lay <= static_cast<int>(tools.lastLayer()); ++lay) {
      for (const int sign : {1, -1}) {
        const auto expected = tools.getPositionLayer(sign * lay);
        const auto got = facade.getPositionLayer(sign * lay);
        if (std::abs(got.z() - expected.z()) >= tol) {
          throw cms::Exception("TICLGeomClosure") << "label '" << label_ << "': getPositionLayer(" << sign * lay
                                                  << ") is " << got.z() << ", RecHitTools " << expected.z();
        }
      }
    }
    for (int lay = 0; lay <= static_cast<int>(tools.lastLayerBarrel()); ++lay) {
      const auto expected = tools.getPositionLayer(lay, false, true);
      const auto got = facade.getPositionLayer(lay, false, true);
      if (std::abs(got.x() - expected.x()) >= tol || std::abs(got.y() - expected.y()) >= tol) {
        throw cms::Exception("TICLGeomClosure")
            << "label '" << label_ << "': barrel getPositionLayer(" << lay << ") is (" << got.x() << ", " << got.y()
            << "), RecHitTools (" << expected.x() << ", " << expected.y() << ")";
      }
    }

    edm::LogPrint("TICLGeomAnalyzer") << "label '" << label_ << "': " << n
                                      << " cells, ordering, lookups, all columns and scalars verified against "
                                         "RecHitTools";
  }

private:
  const std::string label_;
  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
};

DEFINE_FWK_MODULE(TICLGeomAnalyzer);
