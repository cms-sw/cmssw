#ifndef COREFRAMEWORK_DEPRECORD_H
#define COREFRAMEWORK_DEPRECORD_H
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DepRecord
// 
/**\class DepRecord DepRecord.h FWCore/Framework/test/DepRecord.h

 Description: A test Record that is dependent on DummyRecord

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Jun 22 19:46:32 EDT 2005
// $Id: DepRecord.h,v 1.1 2005/06/23 19:48:45 chrjones Exp $
//

// system include files
#include "boost/mpl/vector.hpp"

// user include files

// forward declarations
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/test/DummyRecord.h"

class DepRecord 
: public edm::eventsetup::DependentRecordImplementation<DepRecord, boost::mpl::vector<DummyRecord> >
{
};

#endif /* COREFRAMEWORK_DEPRECORD_H */
