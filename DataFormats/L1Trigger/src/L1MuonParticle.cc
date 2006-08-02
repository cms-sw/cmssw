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
// $Id: L1MuonParticle.cc,v 1.1 2006/07/26 00:05:40 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

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
//   : ParticleWithCharge( q, p4 ),
   : L1PhysObjectBase( q, p4, aRef )
{
   if( triggerObjectRef().isNonnull() )
   {
      isolated_ = gmtMuonCand()->isol() ;
      mip_ = gmtMuonCand()->mip() ;
   }
}

L1MuonParticle::L1MuonParticle( Charge q,
				const LorentzVector& p4,
				bool isolated,
				bool mip )
   : L1PhysObjectBase( q, p4, L1Ref() ),
     isolated_( isolated ),
     mip_( mip )
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

const L1MuGMTCand*
L1MuonParticle::gmtMuonCand() const
{
   return dynamic_cast< const L1MuGMTCand* >( triggerObjectPtr() ) ;
}

//
// const member functions
//

//
// static member functions
//
