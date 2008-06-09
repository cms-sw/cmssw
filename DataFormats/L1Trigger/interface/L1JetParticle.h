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
// $Id: L1JetParticle.h,v 1.10 2007/11/13 03:07:45 wsun Exp $
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
			const edm::Ref< L1GctJetCandCollection >& aRef ) ;

	 L1JetParticle( const PolarLorentzVector& p4,
			const edm::Ref< L1GctJetCandCollection >& aRef ) ;

         // Creates null Ref.
         L1JetParticle( const LorentzVector& p4,
                        JetType type = kUndefined ) ;

         L1JetParticle( const PolarLorentzVector& p4,
                        JetType type = kUndefined ) ;

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

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------
	 void setType( JetType type )
	 { type_ = type ; }

      private:
	 // L1JetParticle(const L1JetParticle&); // stop default

	 // const L1JetParticle& operator=(const L1JetParticle&); // stop default

	 // ---------- member data --------------------------------
         JetType type_ ;
	 edm::Ref< L1GctJetCandCollection > ref_ ;
   };
}

#endif
