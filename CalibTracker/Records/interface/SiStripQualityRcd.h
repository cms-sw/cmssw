#ifndef Records_SiStripQualityRcd_h
#define Records_SiStripQualityRcd_h
// -*- C++ -*-
//
// Package:     Records
// Class  :     SiStripQualityRcd
// 
/**\class SiStripQualityRcd SiStripQualityRcd.h CalibTracker/Records/interface/SiStripQualityRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Domenico Giordano
// Created:     Wed Sep 26 17:42:12 CEST 2007
// $Id: SiStripQualityRcd.h,v 1.2 2007/11/19 15:38:28 giordano Exp $
//

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

class SiStripQualityRcd : public edm::eventsetup::DependentRecordImplementation<SiStripQualityRcd, boost::mpl::vector<SiStripBadModuleRcd, SiStripBadFiberRcd, SiStripBadChannelRcd, SiStripBadStripRcd, SiStripDetCablingRcd, SiStripDCSStatusRcd> > {};

#endif
