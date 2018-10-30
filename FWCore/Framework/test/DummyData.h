#ifndef Framework_DummyData_h
#define Framework_DummyData_h
/*
 *  DummyData.h
 *  EDMProto
 *
 *  Created by Chris Jones on 4/4/05.
 *
 */

//used to set the default record
#include "FWCore/Framework/test/DummyRecord.h"

namespace edm::eventsetup::test {
  struct DummyData { int value_;
    DummyData(int iValue=0) : value_(iValue) {}
    void dummy() {} // Just to suppress compilation warning message
  };
}

#include "FWCore/Framework/interface/data_default_record_trait.h"
EVENTSETUP_DATA_DEFAULT_RECORD(edm::eventsetup::test::DummyData, DummyRecord)

#endif
