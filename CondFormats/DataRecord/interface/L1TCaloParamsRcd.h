///
/// \class L1TCaloParamsRcd
///
/// Description: Record for CaloParams
///
/// Implementation:
///    
///
/// \author: Jim Brooke, University of Bristol
///
#ifndef CondFormatsDataRecord_L1TCaloParamsRcd_h
#define CondFormatsDataRecord_L1TCaloParamsRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class L1TCaloParamsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TCaloParamsRcd> {};

// Dependent record implmentation:
//#include "FWCore/Framework/interface/DependentRecordImplementation.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
//class L1TCaloParamsRcd : public edm::eventsetup::DependentRecordImplementation<L1TCaloParamsRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
