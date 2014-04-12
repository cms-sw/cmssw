#ifndef RECORDS_HCALALIGNMENTRCD_H
#define RECORDS_HCALALIGNMENTRCD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     HcalAlignmentRcd
// 
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentRcd.h"
#include "boost/mpl/vector.hpp"


class HcalAlignmentRcd : 
   public edm::eventsetup::DependentRecordImplementation<
   HcalAlignmentRcd,
		boost::mpl::vector<
                HBAlignmentRcd,
                HOAlignmentRcd,
                HEAlignmentRcd,
                HFAlignmentRcd      > > {};

#endif /* RECORDS_HCALALIGNMENTRCD_H */

