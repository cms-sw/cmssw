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
// $Id$
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/ParticleWithCharge.h"
#include "DataFormats/Common/interface/Ref.h"

// forward declarations
class L1GmtExtendedCandCollection ;

namespace level1 {

   class L1MuonParticle : public ParticleWithCharge
   {

      public:
	 L1MuonParticle();
	 virtual ~L1MuonParticle();

	 // ---------- const member functions ---------------------
	 const edm::Ref< L1GmtMuonCand >& gmtMuonCand() const
	 { return m_gmtMuonCand ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 L1MuonParticle(const L1MuonParticle&); // stop default

	 const L1MuonParticle& operator=(const L1MuonParticle&); // stop default

	 // ---------- member data --------------------------------
	 edm::Ref< L1GmtExtendedCandCollection > m_gmtMuonCand ;
   };
}

#endif
