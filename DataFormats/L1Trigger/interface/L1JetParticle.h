#ifndef L1Trigger_L1JetParticle_h
#define L1Trigger_L1JetParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1JetParticle
// 
/**\class L1JetParticle L1JetParticle.h DataFormats/L1Trigger/interface/L1JetParticle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1JetParticle.h,v 1.4 2006/08/06 15:32:44 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
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

         // Creates null Ref.
         L1JetParticle( const LorentzVector& p4,
                        JetType type = kUndefined ) ;

	 virtual ~L1JetParticle();

	 // ---------- const member functions ---------------------
         JetType type() const
         { return type_ ; }

	 const edm::Ref< L1GctJetCandCollection >& gctJetCandRef() const
	 { return ref_ ; }

	 const L1GctJetCand* gctJetCand() const
	 { return ref_.get() ; }

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
