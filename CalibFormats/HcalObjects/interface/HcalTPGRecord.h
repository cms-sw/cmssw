#ifndef HcalObjects_HcalTPGRecord_h
#define HcalObjects_HcalTPGRecord_h
// -*- C++ -*-
//
// Package:     HcalObjects
// Class  :     HcalTPGRecord
// 
/**\class HcalTPGRecord HcalTPGRecord.h CalibFormats/HcalObjects/interface/HcalTPGRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Thu Sep 14 11:54:26 CDT 2006
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalTPGRecord : public edm::eventsetup::DependentRecordImplementation<HcalTPGRecord, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord,HcalDbRecord> >{};

#endif
