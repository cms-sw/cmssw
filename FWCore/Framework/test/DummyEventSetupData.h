#ifndef EVENTSETUP_TEST_DUMMYEVENTSETUPDATA_H
#define EVENTSETUP_TEST_DUMMYEVENTSETUPDATA_H
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
EVENTSETUP_DATA_DEFAULT_RECORD(edm::DummyEventSetupData, edm::DummyEventSetupRecord);

//NOTE: This should really be put into a .cc file
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
EVENTSETUP_DATA_REG(edm::DummyEventSetupData);

#endif /* EVENTSETUP_TEST_DUMMYEVENTSETUPDATA_H */

