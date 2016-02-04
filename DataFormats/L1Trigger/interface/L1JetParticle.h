#ifndef L1Trigger_L1JetParticle_h
#define L1Trigger_L1JetParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1JetParticle
// 
/**\class L1JetParticle \file L1JetParticle.h DataFormats/L1Trigger/interface/L1JetParticle.h \author Werner Sun

 Description: L1Extra particle class for jet objects.
*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1JetParticle.h,v 1.12 2008/04/03 03:37:20 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/Common/interface/Ref.h"

// forward declarations

namespace l1extra {

   class L1JetParticle : public reco::LeafCandidate
   {

      public:
         enum JetType
         {
            kCentral,
            kForward,
            kTau,
	    kUndefined,
            kNumOfJetTypes
         } ;

	 L1JetParticle();

	 L1JetParticle( const LorentzVector& p4,
			const edm::Ref< L1GctJetCandCollection >& aRef,
			int bx = 0 ) ;

	 L1JetParticle( const PolarLorentzVector& p4,
			const edm::Ref< L1GctJetCandCollection >& aRef,
			int bx = 0 ) ;

         // Creates null Ref.
         L1JetParticle( const LorentzVector& p4,
                        JetType type = kUndefined,
			int bx = 0 ) ;

         L1JetParticle( const PolarLorentzVector& p4,
                        JetType type = kUndefined,
			int bx = 0 ) ;

	 virtual ~L1JetParticle() {}

	 // ---------- const member functions ---------------------
         JetType type() const
         { return type_ ; }

	 const edm::Ref< L1GctJetCandCollection >& gctJetCandRef() const
	 { return ref_ ; }

	 const L1GctJetCand* gctJetCand() const
	 { return ref_.get() ; }

         virtual L1JetParticle* clone() const
         { return new L1JetParticle( *this ) ; }

	 int bx() const
	 { return bx_ ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------
	 void setType( JetType type )
	 { type_ = type ; }

	 void setBx( int bx )
	 { bx_ = bx ; }

      private:
	 // L1JetParticle(const L1JetParticle&); // stop default

	 // const L1JetParticle& operator=(const L1JetParticle&); // stop default

	 // ---------- member data --------------------------------
         JetType type_ ;
	 edm::Ref< L1GctJetCandCollection > ref_ ;
	 int bx_ ;
   };
}

#endif
