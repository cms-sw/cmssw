// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1MuonParticle
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 17:51:21 EDT 2006
// $Id$
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

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
L1MuonParticle::L1MuonParticle()
{
}

L1MuonParticle::L1MuonParticle( Charge q,
				const LorentzVector& p4,
				const L1Ref& aRef )
   : ParticleWithCharge( q, p4 ),
     L1PhysObjectBase( aRef, kMuon )
{
}

// L1MuonParticle::L1MuonParticle(const L1MuonParticle& rhs)
// {
//    // do actual copying here;
// }

L1MuonParticle::~L1MuonParticle()
{
}

//
// assignment operators
//
// const L1MuonParticle& L1MuonParticle::operator=(const L1MuonParticle& rhs)
// {
//   //An exception safe implementation is
//   L1MuonParticle temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

const L1MuGMTExtendedCand*
L1MuonParticle::gmtMuonCand() const
{
   return dynamic_cast< const L1MuGMTExtendedCand* >( triggerObjectPtr() ) ;
}

//
// const member functions
//

//
// static member functions
//
