#ifndef RECORDS_ZDCGEOMETRYRECORD_H
#define RECORDS_ZDCGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     ZDCGeometryRecord
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/Records/interface/PZdcRcd.h"
#include "boost/mpl/vector.hpp"


class ZDCGeometryRecord : 
   public edm::eventsetup::DependentRecordImplementation<
   ZDCGeometryRecord,
		boost::mpl::vector<
                IdealGeometryRecord,
		ZDCAlignmentRcd, 
		ZDCAlignmentErrorRcd,
                ZDCAlignmentErrorExtendedRcd,
		GlobalPositionRcd,
		PZdcRcd         	> > {};

#endif /* RECORDS_ZDCGEOMETRYRECORD_H */

