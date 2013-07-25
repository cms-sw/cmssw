/** \file
 *
 *  $date   : 13/01/2006 16:43:13 CET $
 *  $Revision: 1.2 $
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
DTLayer::DTLayer(DTLayerId id,
                 ReferenceCountingPointer<BoundPlane>& plane,
                 const DTTopology& topo,
                 const DTLayerType& type,
                 const DTSuperLayer* sl) :
  GeomDetUnit(*&plane), theId(id) , theTopo(topo), theType(type) , theSL(sl){
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

const DTSuperLayer* DTLayer::superLayer() const {
  return theSL;
}

const DTChamber* DTLayer::chamber() const {
  return (theSL) ? theSL->chamber() : 0;
}

std::vector< const GeomDet*> DTLayer::components() const {
  return std::vector< const GeomDet*>();
}
