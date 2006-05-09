/** \file GlobalTrackingGeometryBuilder.cc
 * 
 *  $Date: 2006/05/06 13:46:16 $
 *  $Revision: 1.1 $
 *  \author Matteo Sani
 */
 
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

