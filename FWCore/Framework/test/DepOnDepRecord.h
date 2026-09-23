#ifndef Framework_DepOnDepRecord_h
#define Framework_DepOnDepRecord_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DepOnDepRecord
//
/**\class DepOnDepRecord DepOnDepRecord.h FWCore/Framework/test/DepOnDepRecord.h

 Description: A test Record that is dependent on DepRecord

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Jun 22 19:46:32 EDT 2005
//

// system include files

// user include files

// forward declarations
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/test/DepRecord.h"

class DepOnDepRecord
    : public edm::eventsetup::DependentRecordImplementation<DepOnDepRecord, edm::mpl::Vector<DepRecord> > {};

#endif
