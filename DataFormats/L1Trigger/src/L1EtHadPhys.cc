// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtHadPhys
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
#include "DataFormats/L1Trigger/interface/L1EtHadPhys.h"
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
L1EtHadPhys::L1EtHadPhys()
{
}

L1EtHadPhys::L1EtHadPhys( const L1Ref& aRef,
			  float aEtValue )
   : L1EtBase( aRef, aEtValue )
{
}

// L1EtHadPhys::L1EtHadPhys(const L1EtHadPhys& rhs)
// {
//    // do actual copying here;
// }

L1EtHadPhys::~L1EtHadPhys()
{
}

//
// assignment operators
//
// const L1EtHadPhys& L1EtHadPhys::operator=(const L1EtHadPhys& rhs)
// {
//   //An exception safe implementation is
//   L1EtHadPhys temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

const L1GctEtHad*
L1EtHadPhys::gctEtHad() const
{
   return dynamic_cast< const L1GctEtHad* >( triggerObjectPtr() ) ;
}

//
// const member functions
//

//
// static member functions
//
