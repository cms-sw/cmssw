// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 19:05:26 EDT 2006
// $Id$
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EtBase.h"

using namespace l1extra ;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1EtBase::L1EtBase()
{
}

L1EtBase::L1EtBase( const L1Ref& aRef,
		    float aEtValue )
   : L1PhysObjectBase( aRef, kEtSum ),
     etValue_( aEtValue )
{
}

// L1EtBase::L1EtBase(const L1EtBase& rhs)
// {
//    // do actual copying here;
// }

L1EtBase::~L1EtBase()
{
}

//
// assignment operators
//
// const L1EtBase& L1EtBase::operator=(const L1EtBase& rhs)
// {
//   //An exception safe implementation is
//   L1EtBase temp(rhs);
//   swap(rhs);
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
