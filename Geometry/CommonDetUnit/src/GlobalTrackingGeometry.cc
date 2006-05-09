/** \file GlobalTrackingGeometry.cc
 *
 *  $Date: 2006/05/05 14:23:25 $
 *  $Revision: 1.2 $
 *  \author M. Sani
 */

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

using namespace edm;

GlobalTrackingGeometry::GlobalTrackingGeometry() {}

GlobalTrackingGeometry::~GlobalTrackingGeometry() {}

const GeomDetUnit* GlobalTrackingGeometry::idToDetUnit(DetId id) const {
    
    TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != 0) {
        return tg->idToDetUnit(id);
    }
    else {
        // No Tracking Geometry available
        LogInfo("GeometryCommonDetUnit") << "No Tracking Geometry is available.";
        return 0;
    }  
}


const GeomDet* GlobalTrackingGeometry::idToDet(DetId id) const{
  
    TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != 0) {
        return tg->idToDet(id);
    }
    else {
        // No Tracking Geometry available
        LogInfo("GeometryCommonDetUnit") << "No Tracking Geometry is available.";
        return 0;
    }
}

TrackingGeometry* GlobalTrackingGeometry::slaveGeometry(DetId id) const {  
  
    int idx = id.det()-1;
    if (id.det() == DetId::Muon) {
        
        idx+=id.subdetId()-1;
    }
    
    return theGeometries[idx];
}

const TrackingGeometry::DetTypeContainer& GlobalTrackingGeometry::detTypes() const {

    return DetTypeContainer();
}

const TrackingGeometry::DetUnitContainer& GlobalTrackingGeometry::detUnits() const {

    return DetUnitContainer();
}

const TrackingGeometry::DetContainer& GlobalTrackingGeometry::dets() const {

    return DetContainer();
}

const TrackingGeometry::DetIdContainer& GlobalTrackingGeometry::detUnitIds() const {

    return DetIdContainer();
}

const TrackingGeometry::DetIdContainer& GlobalTrackingGeometry::detIds() const {

    return DetIdContainer();
}
