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
// $Id: L1EtMissParticle.h,v 1.3 2006/08/02 14:22:33 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1PhysObjectBase.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

// forward declarations
class L1GctEtMiss ;
class L1GctEtTotal ;
class L1GctEtHad ;

namespace l1extra {

//   class L1EtMissParticle : public reco::ParticleKinematics,
   class L1EtMissParticle : public L1PhysObjectBase
   {

      public:
	 L1EtMissParticle();

	 // Default Refs are null.
	 L1EtMissParticle( const LorentzVector& p4,
			   const double& etTotal,
			   const double& etHad,
			   const L1Ref& aEtMissRef = L1Ref(),
			   const L1Ref& aEtTotalRef = L1Ref(),
			   const L1Ref& aEtHadRef = L1Ref() ) ;

	 virtual ~L1EtMissParticle();

	 // ---------- const member functions ---------------------
	 const L1GctEtMiss* gctEtMiss() const ;
	 const L1GctEtTotal* gctEtTotal() const ;
	 const L1GctEtHad* gctEtHad() const ;

	 double etMiss() const
	 { return et() ; }

	 const double& etTotal() const
	 { return etTot_ ; }

	 const double& etHad() const
	 { return etHad_ ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1EtMissParticle(const L1EtMissParticle&); // stop default

	 // const L1EtMissParticle& operator=(const L1EtMissParticle&); // stop default

	 // ---------- member data --------------------------------
	 double etTot_ ;
	 double etHad_ ;

	 L1Ref etTotRef_ ;
	 L1Ref etHadRef_ ;
   };
}

#endif
