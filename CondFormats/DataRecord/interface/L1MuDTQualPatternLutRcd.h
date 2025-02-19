#ifndef L1MuDTQualPatternLutRCD_H
#define L1MuDTQualPatternLutRCD_H

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuDTQualPatternLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuDTQualPatternLutRcd> {};
class L1MuDTQualPatternLutRcd : public edm::eventsetup::DependentRecordImplementation<L1MuDTQualPatternLutRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
