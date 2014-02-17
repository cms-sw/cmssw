#ifndef L1Trigger_L1MuonParticle_h
#define L1Trigger_L1MuonParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1MuonParticle
// 
/**\class L1MuonParticle \file L1MuonParticle.h DataFormats/L1Trigger/interface/L1MuonParticle.h \author Werner Sun

 Description: L1Extra particle class for muon objects.
*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1MuonParticle.h,v 1.14 2008/04/03 03:37:20 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

// forward declarations

namespace l1extra {

   class L1MuonParticle : public reco::LeafCandidate
   {

      public:
	 L1MuonParticle();

	 // Eventually, all L1MuGMTCands will be L1MuGMTExtendedCands,
	 // as soon as dictionaries for them exist in
	 // L1Trigger/GlobalMuonTrigger.

	 L1MuonParticle( Charge q,
			 const LorentzVector& p4,
			 const L1MuGMTExtendedCand& aCand,
			 int bx = 0 ) ;

	 L1MuonParticle( Charge q,
			 const PolarLorentzVector& p4,
			 const L1MuGMTExtendedCand& aCand,
			 int bx = 0 ) ;

         // Creates null Ref.
         L1MuonParticle( Charge q,
                         const LorentzVector& p4,
                         bool isolated = false,
                         bool mip = false,
			 bool forward = false,
			 bool rpc = false,
			 unsigned int detector = 0,
			 int bx = 0 ) ;

         L1MuonParticle( Charge q,
                         const PolarLorentzVector& p4,
                         bool isolated = false,
                         bool mip = false,
			 bool forward = false,
			 bool rpc = false,
			 unsigned int detector = 0,
			 int bx = 0 ) ;

	 virtual ~L1MuonParticle() {}

	 // ---------- const member functions ---------------------
         bool isIsolated() const
         { return isolated_ ; }

         bool isMip() const
         { return mip_ ; }

	 bool isForward() const
	 { return forward_ ; }

	 bool isRPC() const
	 { return rpc_ ; }

	 const L1MuGMTExtendedCand& gmtMuonCand() const
	 { return cand_ ; }

	 virtual L1MuonParticle* clone() const
	 { return new L1MuonParticle( *this ) ; }

	 int bx() const
	 { return bx_ ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------
	 void setIsolated( bool isIso )
	 { isolated_ = isIso ; }

	 void setMip( bool isMip )
	 { mip_ = isMip ; }

	 void setForward( bool isForward )
	 { forward_ = isForward ; }

	 void setRPC( bool isRPC )
	 { rpc_ = isRPC ; }

	 void setBx( int bx )
	 { bx_ = bx ; }

      private:
	 // L1MuonParticle(const L1MuonParticle&); // stop default

	 // const L1MuonParticle& operator=(const L1MuonParticle&); // stop default

	 // ---------- member data --------------------------------
         bool isolated_ ;
         bool mip_ ;
	 bool forward_ ;
	 bool rpc_ ;
	 L1MuGMTExtendedCand cand_ ;
	 int bx_ ;
   };
}

#endif
