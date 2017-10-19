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

void GEMRing::add( std::shared_ptr< GEMSuperChamber > sch ) {
  superChambers_.emplace_back(sch);
}
  
std::vector< std::shared_ptr< GeomDet >> GEMRing::components() const {
  std::vector< std::shared_ptr< GeomDet > > result;
  for (auto sch : superChambers_) {
    auto newSch(sch->components());
    result.insert(result.end(), newSch.begin(), newSch.end());
  }
  return result;
}

const std::shared_ptr< GeomDet >
GEMRing::component( DetId id ) const {
  return superChamber(GEMDetId(id.rawId()));
}

const std::shared_ptr< GEMSuperChamber >
GEMRing::superChamber( GEMDetId id ) const {
  if (id.region()!=region_ || id.station()!=station_ || id.ring()!=ring_) return nullptr; // not in this station
  return superChamber(id.chamber());
}

const std::shared_ptr< GEMSuperChamber >
GEMRing::superChamber( int isch ) const {
  for (auto sch : superChambers_) {
    if (sch->id().chamber() == isch) {
      return sch;
    }
  }
  return nullptr;
}
  
const std::vector< std::shared_ptr< GEMSuperChamber >>&
GEMRing::superChambers() const {
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
