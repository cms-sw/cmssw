#ifndef L1Trigger_L1EmParticle_h
#define L1Trigger_L1EmParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EmParticle
// 
/**\class L1EmParticle \file L1EmParticle.h DataFormats/L1Trigger/interface/L1EmParticle.h \author Werner Sun

 Description: L1Extra particle class for EM objects.
*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1EmParticle.h,v 1.12 2008/04/03 03:37:20 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/Common/interface/Ref.h"

// forward declarations

namespace l1extra {

   class L1EmParticle : public reco::LeafCandidate
   {

      public:
         enum EmType
         {
            kIsolated,
            kNonIsolated,
	    kUndefined,
            kNumOfEmTypes
         } ;

	 L1EmParticle();

	 L1EmParticle( const LorentzVector& p4,
		       const edm::Ref< L1GctEmCandCollection >& aRef,
		       int bx = 0 ) ;

	 L1EmParticle( const PolarLorentzVector& p4,
		       const edm::Ref< L1GctEmCandCollection >& aRef,
		       int bx = 0 ) ;

         // Creates null Ref.
         L1EmParticle( const LorentzVector& p4,
                       EmType type = kUndefined,
		       int bx = 0 ) ;

         L1EmParticle( const PolarLorentzVector& p4,
                       EmType type = kUndefined,
		       int bx = 0 ) ;

	 virtual ~L1EmParticle() {}

	 // ---------- const member functions ---------------------
         EmType type() const
         { return type_ ; }

	 const edm::Ref< L1GctEmCandCollection >& gctEmCandRef() const
	 { return ref_ ; }

	 const L1GctEmCand* gctEmCand() const
	 { return ref_.get() ; }

         virtual L1EmParticle* clone() const
         { return new L1EmParticle( *this ) ; }

	 int bx() const
	 { return bx_ ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------
	 void setType( EmType type )
	 { type_ = type ; }

	 void setBx( int bx )
	   { bx_ = bx ; }

      private:
	 // L1EmParticle(const L1EmParticle&); // stop default

	 // const L1EmParticle& operator=(const L1EmParticle&); // stop default

	 // ---------- member data --------------------------------
         EmType type_ ;
	 edm::Ref< L1GctEmCandCollection > ref_ ;
	 int bx_ ;
   };
}

#endif
