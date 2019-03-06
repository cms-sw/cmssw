#ifndef Framework_eventsetuprecord_registration_macro_h
#define Framework_eventsetuprecord_registration_macro_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     eventsetuprecord_registration_macro
//
/**\class eventsetuprecord_registration_macro eventsetuprecord_registration_macro.h FWCore/Framework/interface/eventsetuprecord_registration_macro.h

 Description: CPP macro used to register a new EventSetupRecord into the system

 Usage:
    Special code is needed to 'register' a new class that inherits from EventSetupRecord with they system.  The
macro EVENTSETUP_RECORD_REG is used to create that code.

     Example: You have a new record defined called 'DummyRecord'.  Then to register that record with the system
     you place the line

     EVENTSETUP_RECORD_REG(DummyRecord);

     at 'file scope' (i.e., not within a namespace, class or function) in the .cc file for DummyRecord
*/
//
// Author:      Chris Jones
// Created:     Wed Apr  6 14:33:32 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/HCMethods.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/RecordDependencyRegister.h"

#define EVENTSETUP_RECORD_NAME2(_a_, _b_) EVENTSETUP_RECORD_NAME2_HIDDEN(_a_, _b_)
#define EVENTSETUP_RECORD_NAME2_HIDDEN(_a_, _b_) _a_##_b_

#define EVENTSETUP_RECORD_REG(_recordclassname_) \
  TYPELOOKUP_DATA_REG(_recordclassname_);        \
  static const edm::eventsetup::RecordDependencyRegister<_recordclassname_> EVENTSETUP_RECORD_NAME2(s_factory, __LINE__)

#endif
