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
#ifndef CondFormatsDataRecord_L1TCaloParamsO2ORcd_h
#define CondFormatsDataRecord_L1TCaloParamsO2ORcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"
class L1TCaloParamsO2ORcd : public edm::eventsetup::DependentRecordImplementation<L1TCaloParamsO2ORcd, boost::mpl::vector<L1TriggerKeyListExtRcd,L1TriggerKeyExtRcd,L1TCaloStage2ParamsRcd> > {};


#endif
