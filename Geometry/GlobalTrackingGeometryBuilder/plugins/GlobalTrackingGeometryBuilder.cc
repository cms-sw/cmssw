/** \file GlobalTrackingGeometryBuilder.cc
 * 
 *  $Date: 2011/08/16 14:54:34 $
 *  $Revision: 1.1 $
 *  \author Matteo Sani
 */
 
#include <Geometry/GlobalTrackingGeometryBuilder/plugins/GlobalTrackingGeometryBuilder.h>
#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>

#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>


#include <iostream>
#include <iomanip>

GlobalTrackingGeometryBuilder::GlobalTrackingGeometryBuilder() : myName("GlobalTrackingGeometryBuilder"){}


GlobalTrackingGeometryBuilder::~GlobalTrackingGeometryBuilder(){}


GlobalTrackingGeometry* GlobalTrackingGeometryBuilder::build(const TrackerGeometry* tk, 
    const DTGeometry* dt, const CSCGeometry* csc, const RPCGeometry* rpc){

    // DO NOT CHANGE THE ORDER OF THE GEOMETRIES !!!!!!!  
    
    std::vector<const TrackingGeometry*> tkGeometries;
    
    tkGeometries.push_back(tk);
    tkGeometries.push_back(dt);
    tkGeometries.push_back(csc);
    tkGeometries.push_back(rpc);
    
    
    return new GlobalTrackingGeometry(tkGeometries);
}

