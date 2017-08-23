#include <utility>

#include "Geometry/GEMGeometry/interface/GEMStation.h"
#include "Geometry/GEMGeometry/interface/GEMRing.h"
#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"

GEMStation::GEMStation(int region, int station) :
  region_(region),
  station_(station)
{
}

GEMStation::~GEMStation() {}

std::vector<GEMDetId> GEMStation::ids() const {
  std::vector<GEMDetId> result;
  for (auto& ri : rings_){
    std::vector<GEMDetId> newIds(ri->ids());
    result.insert(result.end(), newIds.begin(), newIds.end());
  }
  return result;
}

bool GEMStation::operator==(const GEMStation& st) const {
  return (region_ == st.region() && station_ == st.station());
}

void GEMStation::add(GEMRing* ring) {
  rings_.emplace_back(ring);
}

std::vector<const GeomDet*> GEMStation::components() const {
  std::vector<const GeomDet*> result;
  for (auto ri : rings_) {
    auto newSch(ri->components());
    result.insert(result.end(), newSch.begin(), newSch.end());
  }
  return result;
}

const GeomDet* GEMStation::component(DetId id) const {
 auto detId(GEMDetId(id.rawId()));
 return ring(detId.ring())->component(id);
}

const GEMSuperChamber* GEMStation::superChamber(GEMDetId id) const {
  if (id.region()!=region_ || id.station()!=station_ ) return nullptr; // not in this station
  return ring(id.ring())->superChamber(id.chamber());
}

std::vector<const GEMSuperChamber*> GEMStation::superChambers() const {
  std::vector<const GEMSuperChamber*> result;
  for (auto ri : rings_ ){
    std::vector<const GEMSuperChamber*> newSch(ri->superChambers());
    result.insert(result.end(), newSch.begin(), newSch.end());
  }
  return result;
}

const GEMRing* GEMStation::ring(int ring) const {
  for (auto ri : rings_) {
    if (ring == ri->ring()) {
      return ri;
    }
  }
  return nullptr;
}

const std::vector<const GEMRing*>& GEMStation::rings() const {
  return rings_;
}

int GEMStation::nRings() const {
  return rings_.size();
}

void GEMStation::setName(std::string name) {
  name_ = std::move(name);
}

const std::string GEMStation::getName() const {
  return name_;
}

int GEMStation::region() const {
  return region_;
}

int GEMStation::station() const {
  return station_;
}

