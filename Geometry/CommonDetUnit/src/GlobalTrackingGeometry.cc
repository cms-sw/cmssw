/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "GlobalTrackingGeometry.h"

GlobalTrackingGeometry::GlobalTrackingGeometry(){}

GlobalTrackingGeometry::~GlobalTrackingGeometry(){}



const GeomDetUnit*
idToDetUnit(DetId) const {
  //  return dynamic_cast<const GeomDetUnit*>(idToDet(id));
}


const GeomDet*
idToDet(DetId id) const{
  // FIXME: Check non-zero pointer!
  return slaveGeometry(id).idToDet(id);
}


TrackingGeometry* slaveGeometry(DetId id) {  
  int idx = id.det()-1;
  if (detector == Muon) {
    idx+=id.subdetId()-1;
  }
  return theGeometries[idx];
}
