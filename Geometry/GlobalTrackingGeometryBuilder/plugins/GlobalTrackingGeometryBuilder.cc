/** \file GlobalTrackingGeometryBuilder.cc
 * 
 *  $Date: 2013/05/24 07:43:59 $
 *  $Revision: 1.2 $
 *  \author Matteo Sani
 */
 
#include <Geometry/GlobalTrackingGeometryBuilder/plugins/GlobalTrackingGeometryBuilder.h>
#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>

#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>


#include <iostream>
#include <iomanip>

GlobalTrackingGeometryBuilder::GlobalTrackingGeometryBuilder() : myName("GlobalTrackingGeometryBuilder"){}


GlobalTrackingGeometryBuilder::~GlobalTrackingGeometryBuilder(){}


GlobalTrackingGeometry* GlobalTrackingGeometryBuilder::build(const TrackerGeometry* tk, 
							     const DTGeometry* dt, 
							     const CSCGeometry* csc, 
							     const RPCGeometry* rpc, 
							     const GEMGeometry* gem){

    // DO NOT CHANGE THE ORDER OF THE GEOMETRIES !!!!!!!  
    
    std::vector<const TrackingGeometry*> tkGeometries;
    
    tkGeometries.push_back(tk);
    tkGeometries.push_back(dt);
    tkGeometries.push_back(csc);
    tkGeometries.push_back(rpc);
    tkGeometries.push_back(gem);
    
    
    return new GlobalTrackingGeometry(tkGeometries);
}

