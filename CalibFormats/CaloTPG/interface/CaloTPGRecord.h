#ifndef CaloTPG_CaloTPGRecord_h
#define CaloTPG_CaloTPGRecord_h
// -*- C++ -*-
//
// Package:     CaloTPG
// Class  :     CaloTPGRecord
//
/**\class CaloTPGRecord CaloTPGRecord.h CalibFormats/CaloTPG/interface/CaloTPGRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:
// Created:     Wed Sep 13 19:20:14 CDT 2006
// $Id$
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"

class CaloTPGRecord
    : public edm::eventsetup::DependentRecordImplementation<CaloTPGRecord,
                                                            boost::mpl::vector<HcalLutMetadataRcd, CaloGeometryRecord> > {
};

#endif
