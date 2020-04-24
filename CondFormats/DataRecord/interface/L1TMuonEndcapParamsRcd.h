// Temporary copy to avoid crashes - AWB 28.08.17
// Required to satisfy current convention for "Record" in "Global Tag Entries"
// https://cms-conddb-prod.cern.ch/cmsDbBrowser/search/Prod/L1TMuonEndCapParams

// L1TMuonEndcapParamsRcd                                                                                            
// Description: Record for L1TMuonEndcapParams
//
// automatically generate by make_records.pl
//
#ifndef CondFormatsDataRecord_L1TMuonEndcapParams_h
#define CondFormatsDataRecord_L1TMuonEndcapParams_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class L1TMuonEndcapParamsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TMuonEndcapParamsRcd> {};

// Dependent record implmentation:
//#include "FWCore/Framework/interface/DependentRecordImplementation.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
//class L1TMuonEndcapParamsRcd : public edm::eventsetup::DependentRecordImplementation<L1TMuonEndcapParamsRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
