#ifndef RECORDS_HCALALIGNMENTERRORRCD_H
#define RECORDS_HCALALIGNMENTERRORRCD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     HcalAlignmentErrorRcd
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorRcd.h"
#include "boost/mpl/vector.hpp"


class HcalAlignmentErrorRcd : 
   public edm::eventsetup::DependentRecordImplementation<
   HcalAlignmentErrorRcd,
		boost::mpl::vector<
                HBAlignmentErrorRcd,
                HOAlignmentErrorRcd,
                HEAlignmentErrorRcd,
                HFAlignmentErrorRcd      > > {};

#endif /* RECORDS_HCALALIGNMENTERRORRCD_H */

