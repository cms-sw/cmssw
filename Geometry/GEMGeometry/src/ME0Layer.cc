#include "Geometry/GEMGeometry/interface/ME0Layer.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include <iostream>

ME0Layer::ME0Layer(ME0DetId id, const ReferenceCountingPointer<BoundPlane> & plane) :
  GeomDet(plane), detId_(id)
{
  setDetId(id);
}

ME0Layer::~ME0Layer() {}

ME0DetId ME0Layer::id() const {
  return detId_;
}

bool ME0Layer::operator==(const ME0Layer& ch) const {
  return this->id()==ch.id();
}

void ME0Layer::add(const ME0EtaPartition* rl) {
  etaPartitions_.emplace_back(rl);
}

std::vector<const GeomDet*> ME0Layer::components() const {
  return std::vector<const GeomDet*>(etaPartitions_.begin(), etaPartitions_.end());
}

const GeomDet* ME0Layer::component(DetId id) const {
  return etaPartition(ME0DetId(id.rawId()));
}

const std::vector<const ME0EtaPartition*>& ME0Layer::etaPartitions() const {
  return etaPartitions_;
}

int ME0Layer::nEtaPartitions() const {
  return etaPartitions_.size();
}

const ME0EtaPartition* ME0Layer::etaPartition(ME0DetId id) const {
  if (id.layerId()!=detId_) return nullptr; // not in this eta partition!
  return etaPartition(id.roll());
}

const ME0EtaPartition* ME0Layer::etaPartition(int isl) const {
  for (auto roll : etaPartitions_){
    if (roll->id().roll()==isl) 
      return roll;
  }
  return nullptr;
}
