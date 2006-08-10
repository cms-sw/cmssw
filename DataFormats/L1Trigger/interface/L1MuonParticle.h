#ifndef L1Trigger_L1MuonParticle_h
#define L1Trigger_L1MuonParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1MuonParticle
// 
/**\class L1MuonParticle L1MuonParticle.h DataFormats/L1Trigger/interface/L1MuonParticle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1MuonParticle.h,v 1.4 2006/08/06 15:32:44 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/Common/interface/Ref.h"

// forward declarations

namespace l1extra {

   class L1MuonParticle : public reco::LeafCandidate
   {

      public:
	 L1MuonParticle();

	 L1MuonParticle( Charge q,
			 const LorentzVector& p4,
			 const edm::Ref< std::vector< L1MuGMTCand> >& aRef ) ;

         // Creates null Ref.
         L1MuonParticle( Charge q,
                         const LorentzVector& p4,
                         bool isolated = false,
                         bool mip = false ) ;

	 virtual ~L1MuonParticle();

	 // ---------- const member functions ---------------------
         bool isIsolated() const
         { return isolated_ ; }

         bool isMip() const
         { return mip_ ; }

	 const edm::Ref< std::vector< L1MuGMTCand> >& gmtMuonCandRef() const
	 { return ref_ ; }

	 const L1MuGMTCand* gmtMuonCand() const
	 { return ref_.get() ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------
	 void setIsolated( bool isIso )
	 { isolated_ = isIso ; }

	 void setMip( bool isMip )
	 { mip_ = isMip ; }

      private:
	 // L1MuonParticle(const L1MuonParticle&); // stop default

	 // const L1MuonParticle& operator=(const L1MuonParticle&); // stop default

	 // ---------- member data --------------------------------
         bool isolated_ ;
         bool mip_ ;
	 edm::Ref< std::vector< L1MuGMTCand> > ref_ ;
   };
}

#endif
