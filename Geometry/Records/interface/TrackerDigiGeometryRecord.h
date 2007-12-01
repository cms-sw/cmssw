#ifndef RECORDS_TRACKERDIGIGEOMETRYRECORD_H
#define RECORDS_TRACKERDIGIGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     IdealGeometryRecord
// 
/**\class IdealGeometryRecord IdealGeometryRecord.h Geometry/Records/interface/IdealGeometryRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Mon Jul 25 11:05:09 EDT 2005
// $Id: TrackerDigiGeometryRecord.h,v 1.4 2007/08/13 19:24:09 case Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"
#include "Geometry/Records/interface/PGeometricDetRcd.h"
#include "boost/mpl/vector.hpp"


class TrackerDigiGeometryRecord : 
  public edm::eventsetup::DependentRecordImplementation<TrackerDigiGeometryRecord,
							boost::mpl::vector<IdealGeometryRecord,
									   TrackerAlignmentRcd, 
									   TrackerAlignmentErrorRcd,
		                                                           PGeometricDetRcd> > {};

#endif /* RECORDS_TRACKERDIGIGEOMETRYRECORD_H */

