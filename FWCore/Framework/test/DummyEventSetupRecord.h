#ifndef EVENTSETUP_TEST_DUMMYEVENTSETUPRECORD_H
#define EVENTSETUP_TEST_DUMMYEVENTSETUPRECORD_H
/*
 *  DummyEventSetupRecord.h
 *  EDMProto
 *
 *  Created by Chris Jones on 4/4/05.
 *
 */

#include "FWCore/CoreFramework/interface/EventSetupRecordImplementation.h"

namespace edm {
   class DummyEventSetupRecord : 
     public edm::eventsetup::EventSetupRecordImplementation<DummyEventSetupRecord> {};
}

//NOTE: the following should really go into a DummyEventSetupRecord.cc file
#include "FWCore/CoreFramework/interface/eventsetuprecord_registration_macro.h"
EVENTSETUP_RECORD_REG(edm::DummyEventSetupRecord);

#endif /*EVENTSETUP_TEST_DUMMYEVENTSETUPRECORD_H*/
