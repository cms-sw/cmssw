#ifndef EVENTSETUP_TEST_DUMMYRECORD_H
#define EVENTSETUP_TEST_DUMMYRECORD_H
/*
 *  DummyRecord.h
 *  EDMProto
 *
 *  Created by Chris Jones on 4/4/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class DummyRecord : public edm::eventsetup::EventSetupRecordImplementation<DummyRecord> {};

#endif /*EVENTSETUP_TEST_DUMMYRECORD_H*/
