/** \file
 *
 *  $Date: 2006/05/05 10:12:06 $
 *  $Revision: 1.1 $
 *  \author M. Sani
 */

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>

GlobalTrackingGeometry::GlobalTrackingGeometry() {}

GlobalTrackingGeometry::~GlobalTrackingGeometry() {}

const GeomDetUnit* idToDetUnit(DetId) const {
    
    TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != 0) {
        return dynamic_cast<const GeomDetUnit*>tg->idToDet(id);
    }
    else {
        // No Tracking Geometry available
        LogInfo("GeometryCommonDetUnit") << "No Tracking Geometry is available.";
        return new GeomDetUnit();
    }  
}


const GlobalTrackingGeometry::GeomDet* idToDet(DetId id) const{
  
    TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != 0) {
        return tg->idToDet(id);
    }
    else {
        // No Tracking Geometry available
        LogInfo("GeometryCommonDetUnit") << "No Tracking Geometry is available.";
        return new GeomDetUnit();
    }  
}


TrackingGeometry* GlobalTrackingGeometry::slaveGeometry(DetId id) {  
  
    int idx = id.det()-1;
    if (detector == Muon) {
        
        idx+=id.subdetId()-1;
    }
    
    return theGeometries[idx];
}

const DetTypeContainer& GlobalTrackingGeometry::detTypes() const {

    return new DetTypeContainer();
}

const DetUnitContainer& GlobalTrackingGeometry::detUnits() const {

    return new DetUnitContainer();
}

const DetContainer& GlobalTrackingGeometry::dets() const {

    return new DetContainer();
}

const DetIdContainer& GlobalTrackingGeometry::detUnitIds() const {

    return new DetIdContainer();
}

const DetIdContainer& GlobalTrackingGeometry::detIds() const {

    return new DetIdContainer();
}
