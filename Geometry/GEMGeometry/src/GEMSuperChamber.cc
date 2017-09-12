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

void GEMSuperChamber::add(GEMChamber* ch) {
  chambers_.emplace_back(ch);
}

std::vector<const GeomDet*> GEMSuperChamber::components() const {
  return std::vector<const GeomDet*>(chambers_.begin(), chambers_.end());
}

const GeomDet* GEMSuperChamber::component(DetId id) const {
  return chamber(GEMDetId(id.rawId()));
}

const std::vector<const GEMChamber*>& GEMSuperChamber::chambers() const {
  return chambers_;
}

int GEMSuperChamber::nChambers() const {
  return chambers_.size();
}

const GEMChamber* GEMSuperChamber::chamber(GEMDetId id) const {
  if (id.chamber()!=detId_.chamber()) return nullptr; // not in this super chamber!
  return chamber(id.layer());
}

const GEMChamber* GEMSuperChamber::chamber(int isl) const {
  for (auto ch : chambers_){
    if (ch->id().layer()==isl) 
      return ch;
  }
  return nullptr;
}
