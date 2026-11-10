#include <cmath>
#include <string>

#include "CondFormats/HGCalObjects/interface/alpaka/TICLGeomDevice.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

#include "TICLGeomDeviceCheck.h"

// Checks that the TICLGeom collection reaches the alpaka backend, that every
// cell is found there by its rawDetId, and that every TICLGeomSoAConstView and
// ticlgeom::Cell getter on the device matches ticlgeom::Tools on the host.
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TICLGeomDeviceTest : public stream::EDProducer<> {
  public:
    explicit TICLGeomDeviceTest(const edm::ParameterSet& iConfig)
        : EDProducer(iConfig),
          ticlGeomToken_(esConsumes(iConfig.getParameter<edm::ESInputTag>("src"))),
          hostGeomToken_(consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>(
              iConfig.getParameter<edm::ESInputTag>("src"))) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("src", edm::ESInputTag())->setComment("Label of the TICLGeom device collection");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto const& ticlGeom = iSetup.getData(ticlGeomToken_);
      auto const blocks = ticlGeom.const_view();
      const int32_t n = blocks.common().metadata().size();
      const int32_t nBad = ticlgeomtest::checkLookup(iEvent.queue(), blocks.common(), blocks.lookup());

      if (nBad != 0) {
        throw cms::Exception("TICLGeomDeviceLookup")
            << nBad << " of " << n << " cells failed the device-side indexOf round trip";
      }
      edm::LogPrint("TICLGeomDeviceTest") << n << " cells verified on the device via indexOf and denseIdOf ("
                                          << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) << ")";

      auto const results = ticlgeomtest::runGeomQueries(iEvent.queue(), blocks);

      ticlgeom::Tools hostTools;
      hostTools.setGeometry(iSetup.getData(hostGeomToken_));
      compareCells(hostTools, results.cells.data(), n);
      compareLayers(hostTools, results.layers.data(), results.nLayers);
      compareScalars(hostTools, *results.scalars.data());

      edm::LogPrint("TICLGeomDeviceTest") << n << " cells: every geometry getter on the device matches host Tools ("
                                          << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) << ")";
    }

  private:
    static void require(bool ok, const char* what, uint32_t rawDetId = 0) {
      if (!ok) {
        throw cms::Exception("TICLGeomDeviceTools")
            << "device getter '" << what << "' does not match host Tools for detid " << rawDetId;
      }
    }

    static void requireClose(float a, float b, const char* what, uint32_t rawDetId) {
      constexpr float tol = 1.e-4f;
      require(std::abs(a - b) < tol, what, rawDetId);
    }

    static void requirePosition(ticlgeom::Position const& p,
                                GlobalPoint const& q,
                                const char* what,
                                uint32_t rawDetId = 0) {
      require(p.x == q.x() && p.y == q.y() && p.z == q.z(), what, rawDetId);
    }

    static void compareCells(ticlgeom::Tools const& tools, ticlgeomtest::CellResult const* cells, int32_t n) {
      using namespace ticlgeomtest;
      for (int32_t i = 0; i < n; ++i) {
        auto const& r = cells[i];
        const uint32_t raw = r.rawDetId;
        const DetId id(raw);

        require(r.index == i && r.indexAt == i, "cell index", raw);
        require(tools.denseId(id) == i, "rawDetId", raw);
        requirePosition(r.position, tools.getPosition(id), "position", raw);
        requireClose(r.eta, tools.getEta(id), "eta", raw);
        requireClose(r.etaDisplaced, tools.getEta(id, kVertexZ), "eta displaced", raw);
        requireClose(r.phi, tools.getPhi(id), "phi", raw);
        requireClose(r.pt, tools.getPt(id, kHitEnergy), "pt", raw);
        requireClose(r.ptDisplaced, tools.getPt(id, kHitEnergy, kVertexZ), "pt displaced", raw);
        require(r.zside == tools.zside(id), "zside", raw);
        require(r.layer == static_cast<int32_t>(tools.getLayer(id)), "layer", raw);
        require(r.layerWithOffset == static_cast<int32_t>(tools.getLayerWithOffset(id)), "layerWithOffset", raw);
        require(r.type == tools.getCellType(id), "type", raw);
        require(r.sensorGroup == tools.getSensorGroup(id), "sensorGroup", raw);
        require(r.masked == tools.maskCell(id), "masked", raw);
        require(r.isSilicon == tools.isSilicon(id), "isSilicon", raw);
        require(r.isScintillator == tools.isScintillator(id), "isScintillator", raw);
        require(r.isBarrel == tools.isBarrel(id), "isBarrel", raw);
        require(r.thickness == tools.getSiThickness(id), "thickness", raw);
        require(r.thicknessIndex == tools.getSiThickIndex(id), "thicknessIndex", raw);
        require(r.radiusToSide == tools.getRadiusToSide(id), "radiusToSide", raw);
        require(r.isHalfCell == tools.isHalfCell(id), "isHalfCell", raw);
        require(r.maxIphi == tools.getScintMaxIphi(id), "maxIphi", raw);
        require(r.isFine == tools.isScintillatorFine(id), "isFine", raw);

        const auto wafer = tools.getWafer(id);
        require(r.wafer.u == wafer.first && r.wafer.v == wafer.second, "wafer", raw);
        const auto uv = tools.getCell(id);
        require(r.uv.u == uv.first && r.uv.v == uv.second, "uv", raw);
        const auto dEtaDPhi = tools.getScintDEtaDPhi(id);
        require(r.dEtaDPhi.dEta == dEtaDPhi.first && r.dEtaDPhi.dPhi == dEtaDPhi.second, "dEtaDPhi", raw);
        const auto waferInfo = tools.getWaferInfo(id);
        require(r.waferInfo.type == waferInfo.type && r.waferInfo.partialType == waferInfo.partialType &&
                    r.waferInfo.orientation == waferInfo.orientation &&
                    r.waferInfo.placementIndex == waferInfo.placementIndex &&
                    r.waferInfo.cassette == waferInfo.cassette,
                "waferInfo",
                raw);
        const auto tileInfo = tools.getTileInfo(id);
        require(r.tileInfo.type == tileInfo.type && r.tileInfo.sipm == tileInfo.sipm &&
                    r.tileInfo.cassette == tileInfo.cassette,
                "tileInfo",
                raw);
        require(r.cassette == (r.isScintillator ? tileInfo.cassette : waferInfo.cassette), "cassette", raw);
      }
    }

    static void compareLayers(ticlgeom::Tools const& tools, ticlgeomtest::LayerResult const* layers, int32_t nLayers) {
      for (int32_t l = 0; l < nLayers; ++l) {
        require(layers[l].zPlus == tools.getPositionLayer(l).z(), "layerZ plus");
        require(layers[l].zMinus == tools.getPositionLayer(-l).z(), "layerZ minus");
        require(layers[l].noseZ == tools.getPositionLayer(l, true).z(), "noseLayerZ");
        requirePosition(layers[l].barrel, tools.getPositionLayer(l, false, true), "barrelLayerPosition");
      }
    }

    static void compareScalars(ticlgeom::Tools const& tools, ticlgeomtest::ScalarResult const& r) {
      require(r.nBadInvalid == 0, "default values for invalid input");
      require(r.size == tools.geom().size(), "size");
      require(r.numberOfLayers == static_cast<int32_t>(tools.getNumberOfLayers()), "numberOfLayers");
      require(r.lastLayerEE == static_cast<int32_t>(tools.lastLayerEE()), "lastLayerEE");
      require(r.lastLayerFH == static_cast<int32_t>(tools.lastLayerFH()), "lastLayerFH");
      require(r.firstLayerBH == static_cast<int32_t>(tools.firstLayerBH()), "firstLayerBH");
      require(r.lastLayerBH == static_cast<int32_t>(tools.lastLayerBH()), "lastLayerBH");
      require(r.lastLayerECAL == static_cast<int32_t>(tools.lastLayerECAL()), "lastLayerECAL");
      require(r.lastLayerBarrel == static_cast<int32_t>(tools.lastLayerBarrel()), "lastLayerBarrel");
      require(r.noseLastLayer == static_cast<int32_t>(tools.lastLayer(true)), "noseLastLayer");
      require(r.maxWafersPerLayer == static_cast<int32_t>(tools.maxNumberOfWafersPerLayer()), "maxWafersPerLayer");
      require(r.noseMaxWafersPerLayer == static_cast<int32_t>(tools.maxNumberOfWafersPerLayer(true)),
              "noseMaxWafersPerLayer");
      require(r.scintMaxIphi == tools.getScintMaxIphi(), "scintMaxIphi");
      require(r.geometryType == tools.getGeometryType(), "geometryType");
    }

    device::ESGetToken<TICLGeomDevice, CaloGeometryRecord> ticlGeomToken_;
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> hostGeomToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TICLGeomDeviceTest);
