#ifndef RECORDS_CastorGEOMETRYRECORD_H
#define RECORDS_CastorGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     CastorGeometryRecord
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/CastorAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CastorAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CastorAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/Records/interface/PCastorRcd.h"
#include "boost/mpl/vector.hpp"


class CastorGeometryRecord : 
   public edm::eventsetup::DependentRecordImplementation<
   CastorGeometryRecord,
		boost::mpl::vector<
                IdealGeometryRecord,
		CastorAlignmentRcd, 
		CastorAlignmentErrorRcd,
                CastorAlignmentErrorExtendedRcd,
		GlobalPositionRcd,
		PCastorRcd
		> > {};

#endif /* RECORDS_CastorGEOMETRYRECORD_H */

