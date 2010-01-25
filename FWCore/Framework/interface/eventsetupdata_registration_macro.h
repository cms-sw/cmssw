#ifndef Framework_eventsetupdata_registration_macro_h
#define Framework_eventsetupdata_registration_macro_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     eventsetupdata_registration_macro
// 
/**\class eventsetupdata_registration_macro eventsetupdata_registration_macro.h FWCore/Framework/interface/eventsetupdata_registration_macro.h

 Description: CPP macro used to register a data item to be placed within a EventSetup Record

 Usage:
    Special code is needed to 'register' a new class so that it can be placed within a EventSetup Record. The
macro EVENTSETUP_DATA_REG is used to create that code.

    Example: You have a new data class called 'DummyData'.  Then to register that class with the system you
    place the lines

    #include "<where ever my class decleration lives>/interface/DummyData.h"

    EVENTSETUP_DATA_REG(DummyData);

    into the file <where ever my class decleration lives>/src/T_EventSetup_DummyData.cc

The actual name of the file that uses the 'EVENTSETUP_DATA_REG' macro is not important.  The only important point
the file that uses the 'EVENTSETUP_DATA_REG' macro must be in the same library as the data class it is registering.
*/
//
// Author:      Chris Jones
// Created:     Wed Apr  6 15:21:58 EDT 2005
// $Id: eventsetupdata_registration_macro.h,v 1.10 2010/01/21 15:44:54 chrjones Exp $
//

// system include files

// user include files

#include "FWCore/Utilities/interface/typelookup.h"

#define EVENTSETUP_DATA_REG(_dataclass_) TYPELOOKUP_DATA_REG(_dataclass_)

#endif
