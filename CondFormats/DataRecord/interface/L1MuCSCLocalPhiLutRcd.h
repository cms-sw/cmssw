#ifndef CondFormatsDataRecord_L1MuCSCLocalPhiLutRcd_h
#define CondFormatsDataRecord_L1MuCSCLocalPhiLutRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuCSCLocalPhiLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuCSCLocalPhiLutRcd> {};
class L1MuCSCLocalPhiLutRcd : public edm::eventsetup::DependentRecordImplementation<L1MuCSCLocalPhiLutRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
