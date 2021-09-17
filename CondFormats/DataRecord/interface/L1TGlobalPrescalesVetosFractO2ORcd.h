// L1TGlobalPrescalesVetosFractRcd
// Description: Record for L1TGlobalPrescalesVetosFract
//
// automatically generate by make_records.pl
//
#ifndef CondFormatsDataRecord_L1TGlobalPrescalesVetosFractO2O_h
#define CondFormatsDataRecord_L1TGlobalPrescalesVetosFractO2O_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosFractRcd.h"
class L1TGlobalPrescalesVetosFractO2ORcd
    : public edm::eventsetup::DependentRecordImplementation<
          L1TGlobalPrescalesVetosFractO2ORcd,
          edm::mpl::Vector<L1TriggerKeyListExtRcd, L1TriggerKeyExtRcd, L1TGlobalPrescalesVetosFractRcd> > {};

#endif
