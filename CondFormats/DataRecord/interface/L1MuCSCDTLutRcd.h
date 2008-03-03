#ifndef CondFormatsDataRecord_L1MuCSCDTLutRcd_h
#define CondFormatsDataRecord_L1MuCSCDTLutRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuCSCDTLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuCSCDTLutRcd> {};
class L1MuCSCDTLutRcd : public edm::eventsetup::DependentRecordImplementation<L1MuCSCDTLutRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
