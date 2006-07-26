// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtTotalPhys
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 19:12:37 EDT 2006
// $Id$
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EtTotalPhys.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

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
L1EtTotalPhys::L1EtTotalPhys()
{
}

L1EtTotalPhys::L1EtTotalPhys( const L1Ref& aRef,
			      float aEtValue )
   : L1EtBase( aRef, aEtValue )
{
}

// L1EtTotalPhys::L1EtTotalPhys(const L1EtTotalPhys& rhs)
// {
//    // do actual copying here;
// }

L1EtTotalPhys::~L1EtTotalPhys()
{
}

//
// assignment operators
//
// const L1EtTotalPhys& L1EtTotalPhys::operator=(const L1EtTotalPhys& rhs)
// {
//   //An exception safe implementation is
//   L1EtTotalPhys temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

const L1GctEtTotal*
L1EtTotalPhys::gctEtTotal() const
{
   return dynamic_cast< const L1GctEtTotal* >( triggerObjectPtr() ) ;
}

//
// const member functions
//

//
// static member functions
//
