#ifndef RECORDS_HFGEOMETRYRECORD_H
#define RECORDS_HFGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     HFGeometryRecord
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorRcd.h"
#include "boost/mpl/vector.hpp"


class HFGeometryRecord : 
   public edm::eventsetup::DependentRecordImplementation<
   HFGeometryRecord,
		boost::mpl::vector<
                IdealGeometryRecord,
		HFAlignmentRcd, 
		HFAlignmentErrorRcd> > {};

#endif /* RECORDS_HFGEOMETRYRECORD_H */

