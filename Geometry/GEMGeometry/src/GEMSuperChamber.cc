#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"

GEMSuperChamber::GEMSuperChamber(GEMDetId id, const ReferenceCountingPointer<BoundPlane>& plane)
    : GeomDet(plane), detId_(id) {
  setDetId(id);
}

GEMSuperChamber::~GEMSuperChamber() {}

GEMDetId GEMSuperChamber::id() const { return detId_; }

bool GEMSuperChamber::operator==(const GEMSuperChamber& sch) const { return this->id() == sch.id(); }

void GEMSuperChamber::add(const GEMChamber* ch) { chambers_.emplace_back(ch); }

std::vector<const GeomDet*> GEMSuperChamber::components() const {
  return std::vector<const GeomDet*>(chambers_.begin(), chambers_.end());
}

const GeomDet* GEMSuperChamber::component(DetId id) const { return chamber(GEMDetId(id.rawId())); }

const std::vector<const GEMChamber*>& GEMSuperChamber::chambers() const { return chambers_; }

int GEMSuperChamber::nChambers() const { return chambers_.size(); }

const GEMChamber* GEMSuperChamber::chamber(GEMDetId id) const {
  if (id.chamber() != detId_.chamber())
    return nullptr;  // not in this super chamber!
  return chamber(id.layer());
}

const GEMChamber* GEMSuperChamber::chamber(int isl) const {
  for (auto ch : chambers_) {
    if (ch->id().layer() == isl)
      return ch;
  }
  return nullptr;
}

float GEMSuperChamber::computeDeltaPhi(const LocalPoint& position, const LocalVector& direction) const {
  auto extrap = [](const LocalPoint& point, const LocalVector& dir, double extZ) -> LocalPoint {
    if (dir.z() == 0)
      return LocalPoint(0.f, 0.f, extZ);
    double extX = point.x() + extZ * dir.x() / dir.z();
    double extY = point.y() + extZ * dir.y() / dir.z();
    return LocalPoint(extX, extY, extZ);
  };
  if (nChambers() < 2) {
    return 0.f;
  }

  const float beginOfChamber = chamber(1)->position().z();
  const float centerOfChamber = this->position().z();
  const float endOfChamber = chamber(nChambers())->position().z();

  LocalPoint projHigh =
      extrap(position, direction, (centerOfChamber < 0 ? -1.0 : 1.0) * (endOfChamber - centerOfChamber));
  LocalPoint projLow =
      extrap(position, direction, (centerOfChamber < 0 ? -1.0 : 1.0) * (beginOfChamber - centerOfChamber));
  auto globLow = toGlobal(projLow);
  auto globHigh = toGlobal(projHigh);
  return globHigh.phi() - globLow.phi();  //Geom::phi automatically normalizes to [-pi, pi]
}
