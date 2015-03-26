#ifndef RECORDS_CALOTOWERGEOMETRYRECORD_H
#define RECORDS_CALOTOWERGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     CaloTowerGeometryRecord
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/CaloTowerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CaloTowerAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CaloTowerAlignmentErrorExtendedRcd.h"
#include "Geometry/Records/interface/PCaloTowerRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "boost/mpl/vector.hpp"


class CaloTowerGeometryRecord : 
   public edm::eventsetup::DependentRecordImplementation<
   CaloTowerGeometryRecord,
		boost::mpl::vector<
                IdealGeometryRecord,
		CaloTowerAlignmentRcd, 
		CaloTowerAlignmentErrorRcd,
                CaloTowerAlignmentErrorExtendedRcd,
		GlobalPositionRcd,
                PCaloTowerRcd               > > {};

#endif /* RECORDS_CALOTOWERGEOMETRYRECORD_H */

