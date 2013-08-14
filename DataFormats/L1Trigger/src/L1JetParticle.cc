// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1JetParticle
// 
/**\class L1JetParticle \file L1JetParticle.cc DataFormats/L1Trigger/src/L1JetParticle.cc \author Werner Sun
*/
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 17:51:21 EDT 2006
// $Id: L1JetParticle.cc,v 1.7 2008/04/03 03:37:21 wsun Exp $
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
			      const edm::Ref< L1GctJetCandCollection >& aRef,
			      int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     ref_( aRef ),
     bx_( bx )
{
   if( ref_.isNonnull() )
   {
      type_ = gctJetCand()->isTau() ? kTau :
         ( gctJetCand()->isForward() ? kForward : kCentral ) ;
   }
}

L1JetParticle::L1JetParticle( const PolarLorentzVector& p4,
			      const edm::Ref< L1GctJetCandCollection >& aRef,
			      int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     ref_( aRef ),
     bx_( bx )
{
   if( ref_.isNonnull() )
   {
      type_ = gctJetCand()->isTau() ? kTau :
         ( gctJetCand()->isForward() ? kForward : kCentral ) ;
   }
}

L1JetParticle::L1JetParticle( const LorentzVector& p4,
			      JetType type,
			      int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     type_( type ),
     ref_( edm::Ref< L1GctJetCandCollection >() ),
     bx_( bx )
{
}

L1JetParticle::L1JetParticle( const PolarLorentzVector& p4,
			      JetType type,
			      int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     type_( type ),
     ref_( edm::Ref< L1GctJetCandCollection >() ),
     bx_( bx )
{
}

// L1JetParticle::L1JetParticle(const L1JetParticle& rhs)
// {
//    // do actual copying here;
// }

// L1JetParticle::~L1JetParticle()
// {
// }

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
