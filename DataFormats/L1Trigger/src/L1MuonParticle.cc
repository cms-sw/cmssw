// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1MuonParticle
// 
/**\class L1MuonParticle \file L1MuonParticle.cc DataFormats/L1Trigger/src/L1MuonParticle.cc \author Werner Sun
*/
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 17:51:21 EDT 2006
// $Id: L1MuonParticle.cc,v 1.9 2008/04/03 03:37:21 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

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

L1MuonParticle::L1MuonParticle(
   Charge q,
   const LorentzVector& p4,
   const L1MuGMTExtendedCand& aCand,
   int bx )
  : LeafCandidate( q, p4 ),
    cand_( aCand ),
    bx_( bx )
{
   isolated_ = cand_.isol() ;
   mip_ = cand_.mip() ;
   forward_ = cand_.isFwd() ;
   rpc_ = cand_.isRPC() ;
}

L1MuonParticle::L1MuonParticle(
   Charge q,
   const PolarLorentzVector& p4,
   const L1MuGMTExtendedCand& aCand,
   int bx )
  : LeafCandidate( q, p4 ),
    cand_( aCand ),
    bx_( bx )
{
   isolated_ = cand_.isol() ;
   mip_ = cand_.mip() ;
   forward_ = cand_.isFwd() ;
   rpc_ = cand_.isRPC() ;
}

L1MuonParticle::L1MuonParticle( Charge q,
				const LorentzVector& p4,
				bool isolated,
				bool mip,
				bool forward,
				bool rpc,
				unsigned int detector,
				int bx )
   : LeafCandidate( q, p4 ),
     isolated_( isolated ),
     mip_( mip ),
     forward_( forward ),
     rpc_( rpc ),
     cand_( L1MuGMTExtendedCand() ),
     bx_( bx )
{
}

L1MuonParticle::L1MuonParticle( Charge q,
				const PolarLorentzVector& p4,
				bool isolated,
				bool mip,
				bool forward,
				bool rpc,
				unsigned int detector,
				int bx )
   : LeafCandidate( q, p4 ),
     isolated_( isolated ),
     mip_( mip ),
     forward_( forward ),
     rpc_( rpc ),
     cand_( L1MuGMTExtendedCand() ),
     bx_( bx )
{
}

// L1MuonParticle::L1MuonParticle(const L1MuonParticle& rhs)
// {
//    // do actual copying here;
// }

// L1MuonParticle::~L1MuonParticle()
// {
// }

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

//
// const member functions
//

//
// static member functions
//
