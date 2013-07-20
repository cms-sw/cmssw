#ifndef L1Geometry_L1CaloGeometryRecord_h
#define L1Geometry_L1CaloGeometryRecord_h
// -*- C++ -*-
//
// Package:     L1Geometry
// Class  :     L1CaloGeometryRecord
// 
/**\class L1CaloGeometryRecord L1CaloGeometryRecord.h L1TriggerConfig/L1Geometry/interface/L1CaloGeometryRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Mon Oct 23 23:08:26 EDT 2006
// $Id: L1CaloGeometryRecord.h,v 1.1 2008/04/16 23:06:56 wsun Exp $
//

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

/* class L1CaloGeometryRecord : */
/*    public edm::eventsetup::EventSetupRecordImplementation<L1CaloGeometryRecord> */
/* {}; */
class L1CaloGeometryRecord : public edm::eventsetup::DependentRecordImplementation<L1CaloGeometryRecord, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
