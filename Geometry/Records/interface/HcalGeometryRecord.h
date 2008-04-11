#ifndef RECORDS_HCALGEOMETRYRECORD_H
#define RECORDS_HCALGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     HcalGeometryRecord
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HBGeometryRecord.h"
#include "Geometry/Records/interface/HEGeometryRecord.h"
#include "Geometry/Records/interface/HOGeometryRecord.h"
#include "Geometry/Records/interface/HFGeometryRecord.h"
#include "boost/mpl/vector.hpp"


class HcalGeometryRecord : 
   public edm::eventsetup::DependentRecordImplementation<
   HcalGeometryRecord,
		boost::mpl::vector<
                IdealGeometryRecord,
		HBGeometryRecord, 
		HEGeometryRecord, 
		HOGeometryRecord, 
		HFGeometryRecord > > {};

#endif /* RECORDS_HCALGEOMETRYRECORD_H */

