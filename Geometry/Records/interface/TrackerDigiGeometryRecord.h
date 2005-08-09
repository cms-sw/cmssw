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
// $Id: IdealGeometryRecord.h,v 1.1 2005/07/25 15:11:22 chrjones Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"


class TrackerDigiGeometryRecord : public edm::eventsetup::DependentRecordImplementation<TrackerDigiGeometryRecord,
  boost::mpl::vector<IdealGeometryRecord> > {};

#endif /* RECORDS_TRACKERDIGIGEOMETRYRECORD_H */

