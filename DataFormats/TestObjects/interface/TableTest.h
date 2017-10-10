#ifndef DataFormats_TestObjects_TableTest_h
#define DataFormats_TestObjects_TableTest_h
// -*- C++ -*-
//
// Package:     DataFormats/TestObjects
// Class  :     TableTest
// 
/**\class TableTest TableTest.h "DataFormats/TestObjects/interface/TableTest.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Thu, 31 Aug 2017 15:09:27 GMT
//

// system include files
#include <string>

// user include files
#include "FWCore/SOA/interface/Column.h"
#include "FWCore/SOA/interface/Table.h"

// forward declarations
namespace edmtest {
  //In an actual use of edm::soa::Table one would
  // define the columns in a place that allows sharing
  // of the definitions across many Tables

  SOA_DECLARE_COLUMN(AnInt, int, "anInt");
  SOA_DECLARE_COLUMN(AFloat, float, "aFloat");
  SOA_DECLARE_COLUMN(AString, std::string, "aString");

  using TableTest = edm::soa::Table<AnInt,AFloat,AString>;
}


#endif
