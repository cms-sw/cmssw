#ifndef Geometry_Record_HcalSimNumberingRecord_h
#define Geometry_Record_HcalSimNumberingRecord_h
// -*- C++ -*-
//
// Package:     Record
// Class  :     HcalSimNumberingRecord
//
/**\class HcalSimNumberingRecord HcalSimNumberingRecord.h Geometry/Record/interface/HcalSimNumberingRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:
// Created:     Thu Dec 24 16:41:02 PDT 2013
//

#include <FWCore/Utilities/interface/mplVector.h>
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class HcalSimNumberingRecord
    : public edm::eventsetup::DependentRecordImplementation<HcalSimNumberingRecord,
                                                            edm::mpl::Vector<IdealGeometryRecord, HcalParametersRcd> > {
};

#endif
