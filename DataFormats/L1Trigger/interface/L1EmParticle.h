#ifndef L1Trigger_L1EmParticle_h
#define L1Trigger_L1EmParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EmParticle
// 
/**\class L1EmParticle L1EmParticle.h DataFormats/L1Trigger/interface/L1EmParticle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1EmParticle.h,v 1.1 2006/07/17 20:35:19 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/ParticleKinematics.h"
#include "DataFormats/L1Trigger/interface/L1PhysObjectBase.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

// forward declarations
class L1GctEmCand ;

namespace l1extra {

   class L1EmParticle : public reco::ParticleKinematics,
			public L1PhysObjectBase
   {

      public:
	 L1EmParticle();
	 L1EmParticle( const LorentzVector& p4,
		       const L1Ref& aRef ) ;
	 virtual ~L1EmParticle();

	 // ---------- const member functions ---------------------
	 const L1GctEmCand* gctEmCand() const ;

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1EmParticle(const L1EmParticle&); // stop default

	 // const L1EmParticle& operator=(const L1EmParticle&); // stop default

	 // ---------- member data --------------------------------
   };
}

#endif
