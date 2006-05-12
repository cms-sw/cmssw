/** \file GlobalTrackingGeometry.cc
 *
 *  $Date: 2006/05/10 18:02:27 $
 *  $Revision: 1.4 $
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

    static DetTypeContainer result;
    return result;
}

const TrackingGeometry::DetUnitContainer& GlobalTrackingGeometry::detUnits() const {

    static DetUnitContainer result;
    return result;
}

const TrackingGeometry::DetContainer& GlobalTrackingGeometry::dets() const {

    static DetContainer result;
    return result;
}

const TrackingGeometry::DetIdContainer& GlobalTrackingGeometry::detUnitIds() const {

    static DetIdContainer result;
    return result;
}

const TrackingGeometry::DetIdContainer& GlobalTrackingGeometry::detIds() const {

    static DetIdContainer result;
    return result;
}
