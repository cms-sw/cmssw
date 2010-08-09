#ifndef Geometry_DisplayGeomRecord_h
#define Geometry_DisplayGeomRecord_h
// -*- C++ -*-
//
// Package:     Geometry
// Class  :     DisplayGeomRecord
// 
/**\class DisplayGeomRecord DisplayGeomRecord.h Fireworks/Geometry/interface/DisplayGeomRecord.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Thu Mar 18 16:19:57 CDT 2010
// $Id: DisplayGeomRecord.h,v 1.1 2010/04/01 21:57:59 chrjones Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

class DisplayGeomRecord : public edm::eventsetup::DependentRecordImplementation<DisplayGeomRecord,boost::mpl::vector<IdealGeometryRecord> > {};
class DisplayTrackingGeomRecord : public edm::eventsetup::DependentRecordImplementation<DisplayTrackingGeomRecord,boost::mpl::vector<GlobalTrackingGeometryRecord> > {};

#endif
