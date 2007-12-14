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
// $Id: SiStripQualityRcd.h,v 1.1 2007/10/08 16:46:22 giordano Exp $
//

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiStripBadModuleRcd.h"
#include "CondFormats/DataRecord/interface/SiStripBadFiberRcd.h"
#include "CondFormats/DataRecord/interface/SiStripBadChannelRcd.h"
#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

class SiStripQualityRcd : public edm::eventsetup::DependentRecordImplementation<SiStripQualityRcd, boost::mpl::vector<SiStripBadModuleRcd, SiStripBadFiberRcd, SiStripBadChannelRcd, SiStripBadStripRcd, SiStripDetCablingRcd> > {};

#endif
