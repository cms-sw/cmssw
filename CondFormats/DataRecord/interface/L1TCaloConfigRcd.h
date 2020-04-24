///
/// \class L1TCaloConfigRcd
///
/// Description: Record for CaloConfig
///
/// Implementation:
///    
///
/// \author: Jim Brooke, University of Bristol
///
#ifndef CondFormatsDataRecord_L1TCaloConfigRcd_h
#define CondFormatsDataRecord_L1TCaloConfigRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class L1TCaloConfigRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TCaloConfigRcd> {};

// Dependent record implmentation:
//#include "FWCore/Framework/interface/DependentRecordImplementation.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
//class L1TCaloConfigRcd : public edm::eventsetup::DependentRecordImplementation<L1TCaloConfigRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
