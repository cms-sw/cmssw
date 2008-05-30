#ifndef CondFormatsDataRecord_L1MuCSCTFAlignmentRcd_h
#define CondFormatsDataRecord_L1MuCSCTFAlignmentRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuCSCTFAlignmentRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuCSCTFAlignmentRcd> {};
class L1MuCSCTFAlignmentRcd : public edm::eventsetup::DependentRecordImplementation<L1MuCSCTFAlignmentRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
