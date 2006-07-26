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
// $Id: L1MuonParticle.h,v 1.1 2006/07/17 20:35:19 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/ParticleWithCharge.h"
#include "DataFormats/L1Trigger/interface/L1PhysObjectBase.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

// forward declarations
class L1MuGMTExtendedCand ;

namespace l1extra {

   class L1MuonParticle : public reco::ParticleWithCharge,
			  public L1PhysObjectBase
   {

      public:
	 L1MuonParticle();
	 L1MuonParticle( Charge q,
			 const LorentzVector& p4,
			 const L1Ref& aRef ) ;
	 virtual ~L1MuonParticle();

	 // ---------- const member functions ---------------------
	 const L1MuGMTExtendedCand* gmtMuonCand() const ;

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1MuonParticle(const L1MuonParticle&); // stop default

	 // const L1MuonParticle& operator=(const L1MuonParticle&); // stop default

	 // ---------- member data --------------------------------
   };
}

#endif
