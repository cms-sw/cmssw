/** 
 * Implementation of the Model for a ME0 Chamber
 *
 *  \author S.Dildick 
 *  \edited by D.Nash
 */

#include "Geometry/GEMGeometry/interface/ME0Chamber.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include <iostream>

ME0Chamber::ME0Chamber(ME0DetId id, const ReferenceCountingPointer<BoundPlane> & plane) :
  GeomDet(plane), detId_(id)
{
  setDetId(id);
}

ME0Chamber::~ME0Chamber() {}

ME0DetId ME0Chamber::id() const {
  return detId_;
}

bool ME0Chamber::operator==(const ME0Chamber& ch) const {
  return this->id()==ch.id();
}

void ME0Chamber::add(ME0EtaPartition* rl) {
  etaPartitions_.push_back(rl);
}

std::vector<const GeomDet*> ME0Chamber::components() const {
  return std::vector<const GeomDet*>(etaPartitions_.begin(), etaPartitions_.end());
}

const GeomDet* ME0Chamber::component(DetId id) const {
  return etaPartition(ME0DetId(id.rawId()));
}

const std::vector<const ME0EtaPartition*>& ME0Chamber::etaPartitions() const {
  return etaPartitions_;
}

int ME0Chamber::nEtaPartitions() const {
  return etaPartitions_.size();
}

const ME0EtaPartition* ME0Chamber::etaPartition(ME0DetId id) const {
  if (id.chamberId()!=detId_) return 0; // not in this eta partition!
  return etaPartition(id.roll());
}

const ME0EtaPartition* ME0Chamber::etaPartition(int isl) const {
  for (auto roll : etaPartitions_){
    if (roll->id().roll()==isl) 
      return roll;
  }
  return 0;
}
