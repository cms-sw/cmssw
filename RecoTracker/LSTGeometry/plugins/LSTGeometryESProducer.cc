#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"

#include "RecoTracker/LSTGeometry/interface/Geometry.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

class LSTGeometryESProducer : public edm::ESProducer {
public:
  LSTGeometryESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<lstgeometry::Geometry> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  double ptCut_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;

  const TrackerGeometry *trackerGeom_ = nullptr;
  const TrackerTopology *trackerTopo_ = nullptr;
};

LSTGeometryESProducer::LSTGeometryESProducer(const edm::ParameterSet &iConfig)
    : ptCut_(iConfig.getParameter<double>("ptCut")) {
  std::string ptCutStr = lst::floatToStr(ptCut_, 1);

  auto cc = setWhatProduced(this, ptCutStr);
  geomToken_ = cc.consumes();
  ttopoToken_ = cc.consumes();
}

void LSTGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("ptCut", 0.8);
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<lstgeometry::Geometry> LSTGeometryESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
  trackerGeom_ = &iRecord.get(geomToken_);
  trackerTopo_ = &iRecord.get(ttopoToken_);

  lstgeometry::Sensors sensors;

  std::array<float, lstgeometry::kBarrelLayers> avg_r_barrel{};
  std::array<float, lstgeometry::kEndcapLayers> avg_z_endcap{};
  std::array<unsigned int, lstgeometry::kBarrelLayers> avg_r_barrel_counter{};
  std::array<unsigned int, lstgeometry::kEndcapLayers> avg_z_endcap_counter{};

  for (auto &det : trackerGeom_->dets()) {
    const DetId detId = det->geographicalId();
    const auto moduleType = trackerGeom_->getDetectorType(detId);

    // TODO: Is there a more straightforward way to only loop through these?
    if (moduleType != TrackerGeometry::ModuleType::Ph2PSP && moduleType != TrackerGeometry::ModuleType::Ph2PSS &&
        moduleType != TrackerGeometry::ModuleType::Ph2SS) {
      continue;
    }

    const unsigned int detid = detId();

    const auto &surface = det->surface();
    const auto &position = surface.position();

    const float rho_cm = position.perp();
    const float z_cm = lstgeometry::roundCoordinate(position.z());
    const float phi_rad = lstgeometry::roundAngle(position.phi());

    const auto subdet = trackerGeom_->geomDetSubDetector(detId.subdetId());
    const auto location =
        GeomDetEnumerators::isBarrel(subdet) ? lstgeometry::Location::barrel : lstgeometry::Location::endcap;
    const auto side = static_cast<lstgeometry::Side>(
        location == lstgeometry::Location::barrel ? static_cast<unsigned int>(trackerTopo_->barrelTiltTypeP2(detId))
                                                  : trackerTopo_->side(detId));
    const unsigned int moduleId = trackerTopo_->module(detId);
    const unsigned int layer = trackerTopo_->layer(detId);
    const unsigned int ring = trackerTopo_->endcapRingP2(detId);

    if (layer == 0 || (location == lstgeometry::Location::barrel && layer > lstgeometry::kBarrelLayers) ||
        (location == lstgeometry::Location::endcap && layer > lstgeometry::kEndcapLayers)) {
      continue;
    }

    if (det->isLeaf()) {
      // Leafs are the sensors
      sensors[detid] = lstgeometry::Sensor(
          detid, moduleType, subdet, location, side, moduleId, layer, ring, rho_cm, z_cm, phi_rad, surface);

      continue;
    }

    if (location == lstgeometry::Location::barrel) {
      avg_r_barrel[layer - 1] += rho_cm;
      avg_r_barrel_counter[layer - 1] += 1;
    } else {
      avg_z_endcap[layer - 1] += std::fabs(z_cm);
      avg_z_endcap_counter[layer - 1] += 1;
    }
  }

  for (size_t i = 0; i < avg_r_barrel.size(); ++i) {
    if (avg_r_barrel_counter[i] > 0)
      avg_r_barrel[i] /= avg_r_barrel_counter[i];
  }
  for (size_t i = 0; i < avg_z_endcap.size(); ++i) {
    if (avg_z_endcap_counter[i] > 0)
      avg_z_endcap[i] /= avg_z_endcap_counter[i];
  }

  auto lstGeometry = std::make_unique<lstgeometry::Geometry>(std::move(sensors), avg_r_barrel, avg_z_endcap, ptCut_);

  return lstGeometry;
}

DEFINE_FWK_EVENTSETUP_MODULE(LSTGeometryESProducer);
