/** \file GlobalTrackingGeometryBuilder.cc
 * 
 *  \author Matteo Sani
 */
 
#include "Geometry/GlobalTrackingGeometryBuilder/plugins/GlobalTrackingGeometryBuilder.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"

#include <iostream>
#include <iomanip>

GlobalTrackingGeometryBuilder::GlobalTrackingGeometryBuilder() : myName("GlobalTrackingGeometryBuilder"){}


GlobalTrackingGeometryBuilder::~GlobalTrackingGeometryBuilder(){}


GlobalTrackingGeometry* GlobalTrackingGeometryBuilder::build(const TrackerGeometry* tk,
							     const MTDGeometry* mtd,
							     const DTGeometry* dt, 
							     const CSCGeometry* csc, 
							     const RPCGeometry* rpc, 
							     const GEMGeometry* gem,
							     const ME0Geometry* me0){

    // DO NOT CHANGE THE ORDER OF THE GEOMETRIES !!!!!!!  
    
    std::vector<const TrackingGeometry*> tkGeometries;
    
    tkGeometries.emplace_back(tk);
    tkGeometries.emplace_back(mtd);
    tkGeometries.emplace_back(dt);
    tkGeometries.emplace_back(csc);
    tkGeometries.emplace_back(rpc);
    tkGeometries.emplace_back(gem);
    tkGeometries.emplace_back(me0);
    
    return new GlobalTrackingGeometry(tkGeometries);
}

