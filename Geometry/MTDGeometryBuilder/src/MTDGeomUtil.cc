#include "Geometry/MTDGeometryBuilder/interface/MTDGeomUtil.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace mtd;

void MTDGeomUtil::setGeometry(const MTDGeometry* geom) { geom_ = geom; }

void MTDGeomUtil::setTopology(const MTDTopology* topo) { topology_ = topo; }

bool MTDGeomUtil::isETL(const DetId& id) const {
  MTDDetId hid{id};
  const auto& subDet = hid.mtdSubDetector();
  if (subDet == 0)
    throw cms::Exception("mtd::MTDGeomUtil") << "DetId " << hid.rawId() << " not in MTD!" << std::endl;
  if (subDet == MTDDetId::MTDType::ETL)
    return true;
  return false;
}

bool MTDGeomUtil::isBTL(const DetId& id) const { return !(isETL(id)); }

// row and column set to 0 by default since they are not needed for BTL
std::pair<LocalPoint, GlobalPoint> MTDGeomUtil::position(const DetId& id, int row, int column) const {
  LocalPoint local_point(0., 0., 0.);
  GlobalPoint global_point(0., 0., 0.);
  if (isBTL(id)) {
    BTLDetId detId{id};
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology_->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("mtd::MTDGeomUtil") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                               << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    global_point = thedet->toGlobal(local_point);
  } else {
    ETLDetId detId{id};
    DetId geoId = detId.geographicalId();
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("mtd::MTDGeomUtil") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                               << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    local_point = LocalPoint(topo.localX(row), topo.localY(column), 0.);
    global_point = thedet->toGlobal(local_point);
  }
  return {local_point, global_point};
}

GlobalPoint MTDGeomUtil::globalPosition(const DetId& id, const LocalPoint& local_point) const {
  auto global_point = GlobalPoint(0., 0., 0.);
  if (isBTL(id)) {
    BTLDetId detId{id};
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology_->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("mtd::MTDGeomUtil") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                               << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());
    auto local_point_sim =
        topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    global_point = thedet->toGlobal(local_point_sim);
  } else {
    ETLDetId detId{id};
    DetId geoId = detId.geographicalId();
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("mtd::MTDGeomUtil") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                               << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    global_point = thedet->toGlobal(local_point);
  }
  return global_point;
}

int MTDGeomUtil::zside(const DetId& id) const {
  const MTDDetId hid(id);
  return hid.zside();
}

unsigned int MTDGeomUtil::layer(const DetId& id) const {
  unsigned int layer(0);
  if (isETL(id)) {
    ETLDetId hid(id);
    layer = hid.nDisc();
  }
  return layer;
}

int MTDGeomUtil::module(const DetId& id) const {
  int module = -1;
  if (isETL(id)) {
    ETLDetId hid(id);
    module = hid.module();
  } else {
    BTLDetId hid(id);
    module = hid.module();
  }
  return module;
}

// returns the local position as a pair (x, y) - for ETL
std::pair<float, float> MTDGeomUtil::pixelInModule(const DetId& id, const int row, const int column) const {
  if (isBTL(id))
    throw cms::Exception("mtd::MTDGeomUtil")
        << "ID: " << std::hex << id.rawId() << " from BTL. This method is for ETL only." << std::endl;
  ETLDetId detId(id);
  DetId geoId = detId.geographicalId();
  const MTDGeomDet* thedet = geom_->idToDet(geoId);
  if (thedet == nullptr)
    throw cms::Exception("mtd::MTDGeomUtil") << "GeographicalID: " << std::hex << geoId.rawId() << " (" << detId.rawId()
                                             << ") is invalid!" << std::dec << std::endl;
  const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
  const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());
  const Local3DPoint local_point(topo.localX(row), topo.localY(column), 0.);
  return topo.pixel(local_point);
}

// returns row and column as a pair (row, col)
std::pair<uint8_t, uint8_t> MTDGeomUtil::pixelInModule(const DetId& id, const LocalPoint& local_point) const {
  if (isETL(id)) {
    ETLDetId detId(id);
    DetId geoId = detId.geographicalId();
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("mtd::MTDGeomUtil") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                               << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());
    const auto& thepixel = topo.pixel(local_point);
    uint8_t row(thepixel.first), col(thepixel.second);
    return std::pair<uint8_t, uint8_t>(row, col);
  } else {
    BTLDetId detId(id);
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology_->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("mtd::MTDGeomUtil") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                               << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());
    auto topo_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    const auto& thepixel = topo.pixel(topo_point);
    uint8_t row(thepixel.first), col(thepixel.second);
    return std::pair<uint8_t, uint8_t>(row, col);
  }
}

int MTDGeomUtil::crystalInModule(const DetId& id) const {
  BTLDetId hid(id);
  return hid.crystal();
}

float MTDGeomUtil::eta(const GlobalPoint& position, const float& vertex_z) const {
  GlobalPoint corrected_position = GlobalPoint(position.x(), position.y(), position.z() - vertex_z);
  return corrected_position.eta();
}

float MTDGeomUtil::eta(const DetId& id, const LocalPoint& local_point, const float& vertex_z) const {
  GlobalPoint position = globalPosition(id, local_point);
  float Eta = eta(position, vertex_z);
  return Eta;
}

float MTDGeomUtil::phi(const GlobalPoint& position) const {
  float phi = (position.x() == 0 && position.y() == 0) ? 0 : atan2(position.y(), position.x());
  return phi;
}

float MTDGeomUtil::phi(const DetId& id, const LocalPoint& local_point) const {
  GlobalPoint position = globalPosition(id, local_point);
  float phi = (position.x() == 0 && position.y() == 0) ? 0 : atan2(position.y(), position.x());
  return phi;
}

float MTDGeomUtil::pt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z) const {
  float Eta = eta(position, vertex_z);
  float pt = hitEnergy / cosh(Eta);
  return pt;
}

float MTDGeomUtil::pt(const DetId& id,
                      const LocalPoint& local_point,
                      const float& hitEnergy,
                      const float& vertex_z) const {
  GlobalPoint position = globalPosition(id, local_point);
  float Eta = eta(position, vertex_z);
  float pt = hitEnergy / cosh(Eta);
  return pt;
}
