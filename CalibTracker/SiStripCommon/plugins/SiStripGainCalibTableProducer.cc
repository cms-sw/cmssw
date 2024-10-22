#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "CalibTracker/SiStripCommon/interface/SiStripOnTrackClusterTableProducerBase.h"

class SiStripGainCalibTableProducer : public SiStripOnTrackClusterTableProducerBase {
public:
  explicit SiStripGainCalibTableProducer(const edm::ParameterSet& params)
      : SiStripOnTrackClusterTableProducerBase(params), m_tkGeomToken{esConsumes<>()}, m_gainToken{esConsumes<>()} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "cluster");
    desc.add<std::string>("doc", "");
    desc.add<bool>("extension", false);
    desc.add<edm::InputTag>("Tracks", edm::InputTag{"generalTracks"});
    descriptions.add("siStripGainCalibTable", desc);
  }

  void fillTable(const std::vector<OnTrackCluster>& clusters,
                 const edm::View<reco::Track>& tracks,
                 nanoaod::FlatTable* table,
                 const edm::EventSetup& iSetup) final;

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> m_tkGeomToken;
  const edm::ESGetToken<SiStripGain, SiStripGainRcd> m_gainToken;

  std::map<DetId, double> m_thicknessMap;
  double thickness(DetId id, const TrackerGeometry* tGeom);
};

namespace {
  bool isFarFromBorder(const TrajectoryStateOnSurface& trajState, uint32_t detId, const TrackerGeometry* tGeom) {
    const auto gdu = tGeom->idToDetUnit(detId);
    if ((!dynamic_cast<const StripGeomDetUnit*>(gdu)) && (!dynamic_cast<const PixelGeomDetUnit*>(gdu))) {
      edm::LogWarning("SiStripGainCalibTableProducer")
          << "DetId " << detId << " does not seem to belong to the tracker";
      return false;
    }
    const auto plane = gdu->surface();
    const auto trapBounds = dynamic_cast<const TrapezoidalPlaneBounds*>(&plane.bounds());
    const auto rectBounds = dynamic_cast<const RectangularPlaneBounds*>(&plane.bounds());

    static constexpr double distFromBorder = 1.0;
    double halfLength = 0.;
    if (trapBounds) {
      halfLength = trapBounds->parameters()[3];
    } else if (rectBounds) {
      halfLength = .5 * gdu->surface().bounds().length();
    } else {
      return false;
    }

    const auto pos = trajState.localPosition();
    const auto posError = trajState.localError().positionError();
    if (std::abs(pos.y()) + posError.yy() >= (halfLength - distFromBorder))
      return false;

    return true;
  }
}  // namespace

double SiStripGainCalibTableProducer::thickness(DetId id, const TrackerGeometry* tGeom) {
  const auto it = m_thicknessMap.find(id);
  if (m_thicknessMap.end() != it) {
    return it->second;
  } else {
    double detThickness = 1.;
    const auto gdu = tGeom->idToDetUnit(id);
    const auto isPixel = (dynamic_cast<const PixelGeomDetUnit*>(gdu) != nullptr);
    const auto isStrip = (dynamic_cast<const StripGeomDetUnit*>(gdu) != nullptr);
    if (!isPixel && !isStrip) {
      edm::LogWarning("SiStripGainCalibTableProducer")
          << "DetId " << id.rawId() << " doesn't seem to belong to the Tracker";
    } else {
      detThickness = gdu->surface().bounds().thickness();
    }
    m_thicknessMap[id] = detThickness;
    return detThickness;
  }
}

void SiStripGainCalibTableProducer::fillTable(const std::vector<OnTrackCluster>& clusters,
                                              const edm::View<reco::Track>& tracks,
                                              nanoaod::FlatTable* table,
                                              const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerGeometry> tGeom = iSetup.getHandle(m_tkGeomToken);
  edm::ESHandle<SiStripGain> stripGains = iSetup.getHandle(m_gainToken);

  std::vector<double> c_localdirx;
  std::vector<double> c_localdiry;
  std::vector<double> c_localdirz;
  std::vector<uint16_t> c_firststrip;
  std::vector<uint16_t> c_nstrips;
  std::vector<bool> c_saturation;
  std::vector<bool> c_overlapping;
  std::vector<bool> c_farfromedge;
  std::vector<int> c_charge;
  std::vector<double> c_path;
  // NOTE only very few types are supported by NanoAOD, but more could be added (to discuss with XPOG / core software)
  // NOTE removed amplitude vector, I don't think it was used anywhere
  std::vector<float> c_gainused;      // NOTE was double
  std::vector<float> c_gainusedTick;  // NOTE was double
  for (const auto clus : clusters) {
    const auto& ampls = clus.cluster->amplitudes();
    const int firstStrip = clus.cluster->firstStrip();
    const int nStrips = ampls.size();
    double prevGain = -1;
    double prevGainTick = -1;
    if (stripGains.isValid()) {
      prevGain = stripGains->getApvGain(firstStrip / 128, stripGains->getRange(clus.det, 1), 1);
      prevGainTick = stripGains->getApvGain(firstStrip / 128, stripGains->getRange(clus.det, 0), 1);
    }
    const unsigned int charge = clus.cluster->charge();
    const bool saturation = std::any_of(ampls.begin(), ampls.end(), [](uint8_t amp) { return amp >= 254; });

    const bool overlapping = (((firstStrip % 128) == 0) || ((firstStrip / 128) != ((firstStrip + nStrips) / 128)) ||
                              (((firstStrip + nStrips) % 128) == 127));
    const auto& trajState = clus.measurement.updatedState();
    const auto trackDir = trajState.localDirection();
    const auto cosine = trackDir.z() / trackDir.mag();
    const auto path = (10. * thickness(clus.det, tGeom.product())) / std::abs(cosine);
    const auto farFromEdge = isFarFromBorder(trajState, clus.det, tGeom.product());
    c_localdirx.push_back(trackDir.x());
    c_localdiry.push_back(trackDir.y());
    c_localdirz.push_back(trackDir.z());
    c_firststrip.push_back(firstStrip);
    c_nstrips.push_back(nStrips);
    c_saturation.push_back(saturation);
    c_overlapping.push_back(overlapping);
    c_farfromedge.push_back(farFromEdge);
    c_charge.push_back(charge);
    c_path.push_back(path);
    c_gainused.push_back(prevGain);
    c_gainusedTick.push_back(prevGainTick);
  }
  // addColumn(table, "localdirx", c_localdirx, "<doc>");
  // addColumn(table, "localdiry", c_localdiry, "<doc>");
  // addColumn(table, "localdirz", c_localdirz, "<doc>");
  // addColumn(table, "firststrip", c_firststrip, "<doc>");
  // addColumn(table, "nstrips", c_nstrips, "<doc>");
  addColumn(table, "saturation", c_saturation, "<doc>");
  addColumn(table, "overlapping", c_overlapping, "<doc>");
  addColumn(table, "farfromedge", c_farfromedge, "<doc>");
  addColumn(table, "charge", c_charge, "<doc>");
  // addColumn(table, "path", c_path, "<doc>");
  // ExtendedCalibTree: also charge/path
  addColumn(table, "gainused", c_gainused, "<doc>");
  addColumn(table, "gainusedTick", c_gainusedTick, "<doc>");
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripGainCalibTableProducer);
