/** 
 * Implementation of the Model for a ME0 Chamber
 *
 *  \author P. Verwilligen 
 */

#include "Geometry/GEMGeometry/interface/ME0Chamber.h"
#include "Geometry/GEMGeometry/interface/ME0Layer.h"
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
  if (id.chamberId()!=detId_) return 0; // not in this chamber!
  return layer(id.layer());
}

const ME0Layer* ME0Chamber::layer(int isl) const {
  for (auto lay : layers_){
    if (lay->id().layer()==isl) 
      return lay;
  }
  return 0;
}


