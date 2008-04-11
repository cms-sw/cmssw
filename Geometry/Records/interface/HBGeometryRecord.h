#ifndef RECORDS_HBGEOMETRYRECORD_H
#define RECORDS_HBGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     HBGeometryRecord
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorRcd.h"
#include "boost/mpl/vector.hpp"


class HBGeometryRecord : 
   public edm::eventsetup::DependentRecordImplementation<
   HBGeometryRecord,
		boost::mpl::vector<
                IdealGeometryRecord,
		HBAlignmentRcd, 
		HBAlignmentErrorRcd> > {};

#endif /* RECORDS_HBGEOMETRYRECORD_H */

