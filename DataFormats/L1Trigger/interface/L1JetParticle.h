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
// $Id: L1JetParticle.h,v 1.1 2006/07/17 20:35:19 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/ParticleKinematics.h"
#include "DataFormats/L1Trigger/interface/L1PhysObjectBase.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

// forward declarations
class L1GctJetCand ;

namespace l1extra {

   class L1JetParticle : public reco::ParticleKinematics,
			 public L1PhysObjectBase
   {

      public:
	 L1JetParticle();
	 L1JetParticle( const LorentzVector& p4,
			const L1Ref& aRef ) ;
	 virtual ~L1JetParticle();

	 // ---------- const member functions ---------------------
	 const L1GctJetCand* gctJetCand() const ;

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1JetParticle(const L1JetParticle&); // stop default

	 // const L1JetParticle& operator=(const L1JetParticle&); // stop default

	 // ---------- member data --------------------------------
   };
}

#endif
