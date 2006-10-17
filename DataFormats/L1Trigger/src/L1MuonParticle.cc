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
// $Id: L1MuonParticle.cc,v 1.3 2006/08/10 18:47:42 wsun Exp $
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
   const edm::Ref< std::vector< L1MuGMTCand> >& aRef )
//   : ParticleWithCharge( q, p4 ),
   : LeafCandidate( q, p4 ),
     ref_( aRef )
{
   if( ref_.isNonnull() )
   {
      isolated_ = gmtMuonCand()->isol() ;
      mip_ = gmtMuonCand()->mip() ;
      forward_ = false ;
      rpc_ = false ;
      detector_ = 0 ;
   }
}

L1MuonParticle::L1MuonParticle( Charge q,
				const LorentzVector& p4,
				bool isolated,
				bool mip,
				bool forward,
				bool rpc,
				unsigned int detector )
   : LeafCandidate( q, p4 ),
     isolated_( isolated ),
     mip_( mip ),
     forward_( forward ),
     rpc_( rpc ),
     detector_( detector ),
     ref_( edm::Ref< std::vector< L1MuGMTCand> >() )
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

//
// const member functions
//

//
// static member functions
//
