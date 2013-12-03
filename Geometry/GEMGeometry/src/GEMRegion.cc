#include "Geometry/GEMGeometry/interface/GEMRegion.h"
#include "Geometry/GEMGeometry/interface/GEMStation.h"
#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"

GEMRegion::GEMRegion(int region) : 
  region_(region)
{
}

GEMRegion::~GEMRegion(){}

std::vector<GEMDetId> GEMRegion::ids() const {
  std::vector<GEMDetId> result;
  for (auto st : stations_ ){
    std::vector<GEMDetId> newIds(st->ids());
    result.insert(result.end(), newIds.begin(), newIds.end());
  }
  return result;
}

bool GEMRegion::operator==(const GEMRegion& re) const {
  return region_ == re.region();
}

void GEMRegion::add(GEMStation* st) {
  stations_.push_back(st);
}
  
std::vector<const GeomDet*> GEMRegion::components() const {
  std::vector<const GeomDet*> result;
  for (auto st : stations_) {
    auto newSch(st->components());
    result.insert(result.end(), newSch.begin(), newSch.end());
  }
  return result;
}

const GeomDet* GEMRegion::component(DetId id) const {
  auto detId(GEMDetId(id.rawId()));
  return station(detId.station())->component(id);
}

const GEMSuperChamber* GEMRegion::superChamber(GEMDetId id) const {
  if (id.region()!=region_ ) return 0; // not in this region
  return station(id.station())->ring(id.ring())->superChamber(id);
}

std::vector<const GEMSuperChamber*> GEMRegion::superChambers() const {
  std::vector<const GEMSuperChamber*> result;
  for (auto st : stations_) {
    std::vector<const GEMSuperChamber*> newSch(st->superChambers());
    result.insert(result.end(), newSch.begin(), newSch.end());
  }
  return result;
}

const GEMStation* GEMRegion::station(int st) const {
  for (auto stat : stations_) {
    if (st == stat->station()) {
      return stat;
    }
  }
  return 0;
}

const std::vector<const GEMStation*>& GEMRegion::stations() const {
  return stations_;
}

int GEMRegion::nStations() const {
  return stations_.size();
}

int GEMRegion::region() const {
  return region_;
}
