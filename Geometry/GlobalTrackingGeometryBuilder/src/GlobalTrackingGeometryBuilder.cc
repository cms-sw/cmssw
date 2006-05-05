#include <Geometry/GlobalTrackingGeometryBuilder/src/GlobalTrackingGeometryBuilder.h>
#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <iomanip>

GlobalTrackingGeometryBuilder::GlobalTrackingGeometryBuilder() : myName("GlobalTrackingGeometryBuilder"){}


GlobalTrackingGeometryBuilder::~GlobalTrackingGeometryBuilder(){}


GlobalTrackingGeometry* GlobalTrackingGeometryBuilder::build(const TrackerGeometry* tk, 
    const DTGeometry* dt, const CSCGeometry* csc, const RPCGeometry* rpc){

    return new GlobalTrackingGeometry();

}

