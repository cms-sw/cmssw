/** 
 * Implementation of the Model for a GEM Chamber
 *
 *  \author S.Dildick 
 */

#include "Geometry/GEMGeometry/interface/GEMChamber.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include <iostream>

GEMChamber::GEMChamber(GEMDetId id, const ReferenceCountingPointer<BoundPlane> & plane) :
  GeomDet(plane), detId_(id)
{
  setDetId(id);
}

GEMChamber::~GEMChamber() {}

GEMDetId GEMChamber::id() const {
  return detId_;
}

bool GEMChamber::operator==(const GEMChamber& ch) const {
  return this->id()==ch.id();
}

void GEMChamber::add(GEMEtaPartition* rl) {
  etaPartitions_.emplace_back(rl);
}

std::vector<const GeomDet*> GEMChamber::components() const {
  return std::vector<const GeomDet*>(etaPartitions_.begin(), etaPartitions_.end());
}

const GeomDet* GEMChamber::component(DetId id) const {
  return etaPartition(GEMDetId(id.rawId()));
}

const std::vector<const GEMEtaPartition*>& GEMChamber::etaPartitions() const {
  return etaPartitions_;
}

int GEMChamber::nEtaPartitions() const {
  return etaPartitions_.size();
}

const GEMEtaPartition* GEMChamber::etaPartition(GEMDetId id) const {
  if (id.chamberId()!=detId_) return nullptr; // not in this eta partition!
  return etaPartition(id.roll());
}

const GEMEtaPartition* GEMChamber::etaPartition(int isl) const {
  for (auto roll : etaPartitions_){
    if (roll->id().roll()==isl) 
      return roll;
  }
  return nullptr;
}
