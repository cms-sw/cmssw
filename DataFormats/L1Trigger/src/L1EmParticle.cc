// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EmParticle
// 
/**\class L1EmParticle \file L1EmParticle.cc DataFormats/L1Trigger/src/L1EmParticle.cc \author Werner Sun
 */
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 15:56:47 EDT 2006
// $Id: L1EmParticle.cc,v 1.7 2008/04/03 03:37:21 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"

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
L1EmParticle::L1EmParticle()
{
}

L1EmParticle::L1EmParticle( const LorentzVector& p4,
			    const edm::Ref< L1GctEmCandCollection >& aRef,
			    int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     ref_( aRef ),
     bx_( bx )
{
   if( ref_.isNonnull() )
   {
      type_ = gctEmCand()->isolated() ? kIsolated : kNonIsolated ;
   }
}

L1EmParticle::L1EmParticle( const PolarLorentzVector& p4,
			    const edm::Ref< L1GctEmCandCollection >& aRef,
			    int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     ref_( aRef ),
     bx_( bx )
{
   if( ref_.isNonnull() )
   {
      type_ = gctEmCand()->isolated() ? kIsolated : kNonIsolated ;
   }
}

L1EmParticle::L1EmParticle( const LorentzVector& p4,
			    EmType type,
			    int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     type_( type ),
     ref_( edm::Ref< L1GctEmCandCollection >() ),
     bx_( bx )
{
}

L1EmParticle::L1EmParticle( const PolarLorentzVector& p4,
			    EmType type,
			    int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     type_( type ),
     ref_( edm::Ref< L1GctEmCandCollection >() ),
     bx_( bx )
{
}

// L1EmParticle::L1EmParticle(const L1EmParticle& rhs)
// {
//    // do actual copying here;
// }

// L1EmParticle::~L1EmParticle()
// {
// }

//
// assignment operators
//
// const L1EmParticle& L1EmParticle::operator=(const L1EmParticle& rhs)
// {
//   //An exception safe implementation is
//   L1EmParticle temp(rhs);
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
