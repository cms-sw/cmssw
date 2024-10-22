#ifndef CondFormatsDataRecord_L1CaloHcalScaleRcd_h
#define CondFormatsDataRecord_L1CaloHcalScaleRcd_h

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//class L1CaloHcalScaleRcd : public edm::eventsetup::EventSetupRecordImplementation<L1CaloHcalScaleRcd> {};
class L1CaloHcalScaleRcd : public edm::eventsetup::DependentRecordImplementation<
                               L1CaloHcalScaleRcd,
                               edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd, CaloGeometryRecord> > {};

#endif
