#ifndef Framework_DummyEventSetupData_h
#define Framework_DummyEventSetupData_h
/*
 *  DummyEventSetupData.h
 *  EDMProto
 *
 *  Created by Chris Jones on 4/4/05.
 *
 */

//used to set the default record
#include "FWCore/Framework/test/DummyEventSetupRecord.h"

namespace edm {
   struct DummyEventSetupData { 
      DummyEventSetupData(int iValue) : value_(iValue) {}
      int value_; 
   };
}

#include "FWCore/Framework/interface/data_default_record_trait.h"
EVENTSETUP_DATA_DEFAULT_RECORD(edm::DummyEventSetupData, edm::DummyEventSetupRecord)

#if !defined(TEST_EXCLUDE_DEF)
//NOTE: This should really be put into a .cc file
#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(edm::DummyEventSetupData);
#endif
#endif

