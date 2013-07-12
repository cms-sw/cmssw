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
// $Id: HcalTPGRecord.h,v 1.2 2007/07/11 22:05:49 mansj Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

class HcalTPGRecord : public edm::eventsetup::DependentRecordImplementation<HcalTPGRecord, boost::mpl::vector<IdealGeometryRecord,HcalDbRecord> >{};

#endif
