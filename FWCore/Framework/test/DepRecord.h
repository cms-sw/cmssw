#ifndef COREFRAMEWORK_DEPRECORD_H
#define COREFRAMEWORK_DEPRECORD_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DepRecord
// 
/**\class DepRecord DepRecord.h FWCore/CoreFramework/test/DepRecord.h

 Description: A test Record that is dependent on DummyRecord

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Jun 22 19:46:32 EDT 2005
// $Id$
//

// system include files
#include "boost/mpl/vector.hpp"

// user include files

// forward declarations
#include "FWCore/CoreFramework/interface/DependentRecordImplementation.h"
#include "FWCore/CoreFramework/test/DummyRecord.h"

class DepRecord 
: public edm::eventsetup::DependentRecordImplementation<DepRecord, boost::mpl::vector<DummyRecord> >
{
};

#endif /* COREFRAMEWORK_DEPRECORD_H */
