#ifndef L1Trigger_L1EtMissParticle_h
#define L1Trigger_L1EtMissParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtMissParticle
// 
/**\class L1EtMissParticle L1EtMissParticle.h DataFormats/L1Trigger/interface/L1EtMissParticle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1EtMissParticle.h,v 1.1 2006/07/17 20:35:19 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/ParticleKinematics.h"
#include "DataFormats/L1Trigger/interface/L1PhysObjectBase.h"

// forward declarations
class L1GctEtMiss ;

namespace l1extra {

   class L1EtMissParticle : public reco::ParticleKinematics,
			    public L1PhysObjectBase
   {

      public:
	 L1EtMissParticle();
	 L1EtMissParticle( const LorentzVector& p4,
			   const L1Ref& aRef ) ;
	 virtual ~L1EtMissParticle();

	 // ---------- const member functions ---------------------
	 const L1GctEtMiss* gctEtMiss() const ;

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1EtMissParticle(const L1EtMissParticle&); // stop default

	 // const L1EtMissParticle& operator=(const L1EtMissParticle&); // stop default

	 // ---------- member data --------------------------------
   };
}

#endif
