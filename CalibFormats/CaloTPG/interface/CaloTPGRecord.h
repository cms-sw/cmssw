#ifndef CalibFormats_CaloTPG_CaloTPGRecord_h
#define CalibFormats_CaloTPG_CaloTPGRecord_h

// -*- C++ -*-
//
// Package:     CaloTPG
// Class  :     CaloTPGRecord
//
/**\class CaloTPGRecord CaloTPGRecord.h CalibFormats/CaloTPG/interface/CaloTPGRecord.h

 Description: Calo TPG coder record to follow changes in HCAL LUT meta data and geometry

 Usage:
    <usage>

*/
//
// Author:
// Created:     Wed Sep 13 19:20:14 CDT 2006
//

#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class CaloTPGRecord
    : public edm::eventsetup::DependentRecordImplementation<CaloTPGRecord,
                                                            edm::mpl::Vector<HcalLutMetadataRcd, CaloGeometryRecord> > {
};

#endif
