#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"

GEMSuperChamber::GEMSuperChamber(GEMDetId id, const ReferenceCountingPointer<BoundPlane> & plane) :
  GeomDet(plane), detId_(id)
{
  setDetId(id);
}

GEMSuperChamber::~GEMSuperChamber() {}

GEMDetId GEMSuperChamber::id() const {
  return detId_;
}

bool GEMSuperChamber::operator==(const GEMSuperChamber& sch) const {
  return this->id()==sch.id();
}

void GEMSuperChamber::add( std::shared_ptr< GEMChamber > ch ) {
  chambers_.emplace_back(ch);
}

std::vector< std::shared_ptr< GeomDet >>
GEMSuperChamber::components() const {
  return std::vector< std::shared_ptr< GeomDet >>( chambers_.begin(), chambers_.end());
}

const std::shared_ptr< GeomDet >
GEMSuperChamber::component( DetId id ) const {
  return chamber( GEMDetId( id.rawId()));
}

const std::vector< std::shared_ptr< GEMChamber >>&
GEMSuperChamber::chambers() const {
  return chambers_;
}

int GEMSuperChamber::nChambers() const {
  return chambers_.size();
}

const std::shared_ptr< GEMChamber >
GEMSuperChamber::chamber( GEMDetId id ) const {
  if (id.chamber()!=detId_.chamber()) return nullptr; // not in this super chamber!
  return chamber( id.layer());
}

const std::shared_ptr< GEMChamber >
GEMSuperChamber::chamber( int isl ) const {
  for (auto ch : chambers_){
    if (ch->id().layer()==isl) 
      return ch;
  }
  return nullptr;
}
