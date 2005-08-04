// -*- C++ -*-
//
// Package:     Framework
// Class  :     IOVSyncValue
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  3 18:35:35 EDT 2005
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/interface/IOVSyncValue.h"


//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {

//
// static data member definitions
//

//
// constructors and destructor
//
//IOVSyncValue::IOVSyncValue()
//{
//}

// IOVSyncValue::IOVSyncValue( const IOVSyncValue& rhs )
// {
//    // do actual copying here;
// }

//IOVSyncValue::~IOVSyncValue()
//{
//}

//
// assignment operators
//
// const IOVSyncValue& IOVSyncValue::operator=( const IOVSyncValue& rhs )
// {
//   //An exception safe implementation is
//   IOVSyncValue temp(rhs);
//   swap( rhs );
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
const IOVSyncValue&
IOVSyncValue::invalidIOVSyncValue() {
   static IOVSyncValue s_invalid(0);
   return s_invalid;
}
const IOVSyncValue&
IOVSyncValue::endOfTime() {
   static IOVSyncValue s_endOfTime(0xFFFFFFFFUL);
   return s_endOfTime;
}
const IOVSyncValue&
IOVSyncValue::beginOfTime() {
   static IOVSyncValue s_beginOfTime(1);
   return s_beginOfTime;
}
   }
}
