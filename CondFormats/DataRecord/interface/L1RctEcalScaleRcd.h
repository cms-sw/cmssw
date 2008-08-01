#ifndef CondFormatsDataRecord_L1RctEcalScaleRcd_h
#define CondFormatsDataRecord_L1RctEcalScaleRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1RctEcalScaleRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RctEcalScaleRcd> {};
class L1RctEcalScaleRcd : public edm::eventsetup::DependentRecordImplementation<L1RctEcalScaleRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
