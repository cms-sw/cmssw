#ifndef CalibFormats_HcalObjects_HcalTPGRecord_h
#define CalibFormats_HcalObjects_HcalTPGRecord_h

// -*- C++ -*-
//
// Package:     CalibFormats/HcalObjects
// Class  :     HcalTPGRecord
//
/**\class HcalTPGRecord HcalTPGRecord.h CalibFormats/HcalObjects/interface/HcalTPGRecord.h

 Description: Record for HCAL TPG coders to follow changes in geometry and conditions

 Usage:
    <usage>

*/
//
// Author:
// Created:     Thu Sep 14 11:54:26 CDT 2006
//

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalTPGRecord : public edm::eventsetup::DependentRecordImplementation<
                          HcalTPGRecord,
                          edm::mpl::Vector<HcalRecNumberingRecord, IdealGeometryRecord, HcalDbRecord> > {};

#endif
