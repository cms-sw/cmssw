#include "Geometry/GEMGeometry/interface/GEMRing.h"
#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"

GEMRing::GEMRing(int region, int station, int ring) : 
  region_(region),
  station_(station),
  ring_(ring)
{}

GEMRing::~GEMRing(){}

std::vector<GEMDetId> GEMRing::ids() const {
  return detIds_;
}

bool GEMRing::operator==(const GEMRing& ri) const {
  return ( region_ == ri.region() && 
	   station_ == ri.station() &&
	   ring_ == ri.ring() );
}

void GEMRing::add(GEMSuperChamber* sch) {
  superChambers_.push_back(sch);
}
  
std::vector<const GeomDet*> GEMRing::components() const {
  std::vector<const GeomDet*> result;
  for (auto sch : superChambers_) {
    auto newSch(sch->components());
    result.insert(result.end(), newSch.begin(), newSch.end());
  }
  return result;
}

const GeomDet* GEMRing::component(DetId id) const {
  return superChamber(GEMDetId(id.rawId()));
}

const GEMSuperChamber* GEMRing::superChamber(GEMDetId id) const {
  if (id.region()!=region_ || id.station()!=station_ || id.ring()!=ring_) return 0; // not in this station
  return superChamber(id.chamber());
}

const GEMSuperChamber* GEMRing::superChamber(int isch) const {
  for (auto sch : superChambers_) {
    if (sch->id().chamber() == isch) {
      return sch;
    }
  }
  return 0;
}
  
std::vector<const GEMSuperChamber*> GEMRing::superChambers() const {
  return superChambers_;
}

int GEMRing::nSuperChambers() const {
  return superChambers_.size();
}

int GEMRing::region() const {
  return region_;
}
  
int GEMRing::station() const {
  return station_;
}
  
int GEMRing::ring() const {
  return ring_;
}
