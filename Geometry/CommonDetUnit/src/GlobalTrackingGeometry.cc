/** \file GlobalTrackingGeometry.cc
 *
 *  $Date: 2006/05/09 13:46:16 $
 *  $Revision: 1.3 $
 *  \author M. Sani
 */

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace edm;

GlobalTrackingGeometry::GlobalTrackingGeometry(std::vector<const TrackingGeometry*>& geos) : theGeometries(geos) {}

GlobalTrackingGeometry::~GlobalTrackingGeometry() {}

const GeomDetUnit* GlobalTrackingGeometry::idToDetUnit(DetId id) const {
    
    const TrackingGeometry* tg = slaveGeometry(id);
    
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
  
    const TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != 0) {
        return tg->idToDet(id);
    }
    else {
        // No Tracking Geometry available
        LogInfo("GeometryCommonDetUnit") << "No Tracking Geometry is available.";
        return 0;
    }
}

const TrackingGeometry* GlobalTrackingGeometry::slaveGeometry(DetId id) const {  
  
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
