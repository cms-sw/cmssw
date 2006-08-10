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
// $Id: L1JetParticle.cc,v 1.2 2006/08/02 14:22:33 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

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
			      const edm::Ref< L1GctJetCandCollection >& aRef )
//   : ParticleKinematics( p4 ),
   : LeafCandidate( ( char ) 0, p4 ),
     ref_( aRef )
{
   if( ref_.isNonnull() )
   {
      type_ = gctJetCand()->isTau() ? kTau :
         ( gctJetCand()->isForward() ? kForward : kCentral ) ;
   }
}

L1JetParticle::L1JetParticle( const LorentzVector& p4,
			      JetType type )
   : LeafCandidate( ( char ) 0, p4 ),
     type_( type ),
     ref_( edm::Ref< L1GctJetCandCollection >() )
     
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

//
// const member functions
//

//
// static member functions
//
