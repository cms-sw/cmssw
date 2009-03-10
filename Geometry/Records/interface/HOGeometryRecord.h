#ifndef RECORDS_HOGEOMETRYRECORD_H
#define RECORDS_HOGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     HOGeometryRecord
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorRcd.h"
#include "boost/mpl/vector.hpp"


class HOGeometryRecord : 
   public edm::eventsetup::DependentRecordImplementation<
   HOGeometryRecord,
		boost::mpl::vector<
                IdealGeometryRecord,
		HOAlignmentRcd, 
		HOAlignmentErrorRcd> > {};

#endif /* RECORDS_HOGEOMETRYRECORD_H */

