///
/// \class L1TYellowParamsRcd
///
/// Description: Record for YellowParams of the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to implment a record.
///
/// \author: Michael Mulhearn - UC Davis
///
#ifndef CondFormatsDataRecord_L1TYellowParamsRcd_h
#define CondFormatsDataRecord_L1TYellowParamsRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class L1TYellowParamsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TYellowParamsRcd> {};

// Dependent record implmentation:
//#include "FWCore/Framework/interface/DependentRecordImplementation.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
//class L1TYellowParamsRcd : public edm::eventsetup::DependentRecordImplementation<L1TYellowParamsRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
