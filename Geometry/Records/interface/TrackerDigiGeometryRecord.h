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
// $Id: TrackerDigiGeometryRecord.h,v 1.1 2005/08/09 10:37:47 tboccali Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "boost/mpl/vector.hpp"


class TrackerDigiGeometryRecord : 
  public edm::eventsetup::DependentRecordImplementation<TrackerDigiGeometryRecord,
														boost::mpl::vector<IdealGeometryRecord,
																		   TrackerAlignmentRcd > > {};

#endif /* RECORDS_TRACKERDIGIGEOMETRYRECORD_H */

