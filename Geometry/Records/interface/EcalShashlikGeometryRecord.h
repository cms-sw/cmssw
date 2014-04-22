#ifndef RECORDS_ECALSHASHLIKGEOMETRYRECORD_H
#define RECORDS_ECALSHASHLIKGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     EcalShashlikGeometryRecord
// 
//
// Author:      Fedor Ratnikov
// Created:     April 1, 2014
//
// Reduced signature from:
/* class EcalShashlikGeometryRecord :  */
/*   public edm::eventsetup::DependentRecordImplementation< */
/*    EcalShashlikGeometryRecord, */
/* 		boost::mpl::vector< */
/*                 IdealGeometryRecord, */
/* 		EKAlignmentRcd,  */
/* 		EKAlignmentErrorRcd, */
/* 		GlobalPositionRcd, */
/*                 PEcalShashlikRcd */
/* 		> > {}; */
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PEcalShashlikRcd.h" 
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h" 
#include "boost/mpl/vector.hpp"


class EcalShashlikGeometryRecord : 
public edm::eventsetup::DependentRecordImplementation <
      EcalShashlikGeometryRecord,
      boost::mpl::vector <
                IdealGeometryRecord,
 		EEAlignmentRcd,  
 		EEAlignmentErrorRcd, 
                GlobalPositionRcd, 
                PEcalShashlikRcd 
      > 
> {};

#endif 

