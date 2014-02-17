#ifndef DataRecord_RunSummaryRcd_h
#define DataRecord_RunSummaryRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     RunSummaryRcd RunInfoRcd
// 
/**\class RunSummaryRcd RunSummaryRcd.h CondFormats/DataRecord/interface/RunSummaryRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Thus 25 Sep 11:20:27 CEST 2008
// $Id: RunSummaryRcd.h,v 1.2 2008/10/15 12:12:11 degrutto Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class RunSummaryRcd : public edm::eventsetup::EventSetupRecordImplementation<RunSummaryRcd> {};

class RunInfoRcd : public edm::eventsetup::EventSetupRecordImplementation<RunInfoRcd> {};

#endif
