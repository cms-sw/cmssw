/** \file
 *
 *  \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* This Class Header */
#include "Geometry/DTGeometry/interface/DTLayer.h"

/* Collaborating Class Header */
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

/* Base Class Headers */

/* C++ Headers */

/* ====================================================================== */

/* Constructor */ 
DTLayer::DTLayer(const DTLayerId& id,
                 ReferenceCountingPointer<BoundPlane>& plane,
                 const DTTopology& topo,
                 const DTLayerType& type,
                 std::shared_ptr< DTSuperLayer > sl) :
  GeomDet(*&plane), theId(id) , theTopo(topo), theType(type) , theSL(sl){
  setDetId(id);
}

/* Destructor */ 
DTLayer::~DTLayer() {
}

/* Operations */ 
const Topology& DTLayer::topology() const {
  return theTopo;
}

const GeomDetType& DTLayer::type() const{
  return theType;
}

const DTTopology& DTLayer::specificTopology() const {
  return theTopo;
}

DTLayerId DTLayer::id() const {
  return theId;
}

bool DTLayer::operator==(const DTLayer& l) const {
  return id()==l.id();
}

const std::shared_ptr< DTSuperLayer >
DTLayer::superLayer() const {
  return theSL;
}

const std::shared_ptr< DTChamber >
DTLayer::chamber() const {
  return (theSL) ? theSL->chamber() : nullptr;
}

std::vector< std::shared_ptr< GeomDet >>
DTLayer::components() const {
  return std::vector< std::shared_ptr< GeomDet >>();
}
