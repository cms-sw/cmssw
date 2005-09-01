#ifndef Framework_DEPRecord_h
#define Framework_DEPRecord_h
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
// $Id: DepRecord.h,v 1.2 2005/07/14 22:50:53 wmtan Exp $
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

#endif /* Framework_DEPRecord_h */
