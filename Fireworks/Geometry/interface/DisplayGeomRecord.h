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
// $Id: DisplayGeomRecord.h,v 1.2 2010/08/09 09:07:21 yana Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class DisplayGeomRecord : public edm::eventsetup::DependentRecordImplementation<DisplayGeomRecord,boost::mpl::vector<IdealGeometryRecord> > {};

#endif
