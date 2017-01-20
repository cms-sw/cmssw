#include "Geometry/GEMGeometry/interface/ME0Chamber.h"
#include "Geometry/GEMGeometry/interface/ME0Layer.h"
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

void ME0Chamber::add(ME0Layer* rl) {
  layers_.push_back(rl);
}

std::vector<const GeomDet*> ME0Chamber::components() const {
  return std::vector<const GeomDet*>(layers_.begin(), layers_.end());
}

const GeomDet* ME0Chamber::component(DetId id) const {
  return layer(ME0DetId(id.rawId()));
}

const std::vector<const ME0Layer*>& ME0Chamber::layers() const {
  return layers_;
}

int ME0Chamber::nLayers() const {
  return layers_.size();
}

const ME0Layer* ME0Chamber::layer(ME0DetId id) const {
  if (id.chamberId()!=detId_) return 0; // not in this layer!
  return layer(id.roll());
}

const ME0Layer* ME0Chamber::layer(int isl) const {
  for (auto roll : layers_){
    if (roll->id().roll()==isl) 
      return roll;
  }
  return 0;
}

// For the old ME0 Geometry (with one eta partition)
// we need to maintain this for a while 
void ME0Chamber::add(ME0EtaPartition* rl) {
  etaPartitions_.push_back(rl);
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
