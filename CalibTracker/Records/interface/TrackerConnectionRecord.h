#ifndef GEOMETRY_RECORDS_TRACKERCONNECTIONRECORD_H
#define GEOMETRY_RECORDS_TRACKERCONNECTIONRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     TrackerConnectionRecord
// 
/**\class TrackerConnectionRecord TrackerConnectionRecord.h CalibTracker/Records/interface/TrackerConnectionRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Wed Aug 10 08:13:43 CEST 2005
// $Id$
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class TrackerConnectionRecord : public edm::eventsetup::DependentRecordImplementation<TrackerConnectionRecord,
  boost::mpl::vector<TrackerDigiGeometryRecord> > {};

#endif /* RECORDS_TRACKERCABLINGRECORD_H */

