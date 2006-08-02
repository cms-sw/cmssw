// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1JetParticle
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 17:51:21 EDT 2006
// $Id: L1JetParticle.cc,v 1.1 2006/07/26 00:05:40 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

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
L1JetParticle::L1JetParticle()
{
}

L1JetParticle::L1JetParticle( const LorentzVector& p4,
			      const L1Ref& aRef )
//   : ParticleKinematics( p4 ),
   : L1PhysObjectBase( ( char ) 0, p4, aRef )
{
   if( triggerObjectRef().isNonnull() )
   {
      type_ = gctJetCand()->isTau() ? kTau :
         ( gctJetCand()->isForward() ? kForward : kCentral ) ;
   }
}

L1JetParticle::L1JetParticle( const LorentzVector& p4,
			      JetType type )
   : L1PhysObjectBase( ( char ) 0, p4, L1Ref() ),
     type_( type )
{
}

// L1JetParticle::L1JetParticle(const L1JetParticle& rhs)
// {
//    // do actual copying here;
// }

L1JetParticle::~L1JetParticle()
{
}

//
// assignment operators
//
// const L1JetParticle& L1JetParticle::operator=(const L1JetParticle& rhs)
// {
//   //An exception safe implementation is
//   L1JetParticle temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

const L1GctJetCand*
L1JetParticle::gctJetCand() const
{
   return dynamic_cast< const L1GctJetCand* >( triggerObjectPtr() ) ;
}

//
// const member functions
//

//
// static member functions
//
