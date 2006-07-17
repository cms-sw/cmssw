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
// $Id$
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/ParticleKinematics.h"
#include "DataFormats/Common/interface/RefProd.h"

// forward declarations
class L1GctEtMiss ;

namespace level1 {

   class L1EtMissParticle : public ParticleKinematics
   {

      public:
	 L1EtMissParticle();
	 virtual ~L1EtMissParticle();

	 // ---------- const member functions ---------------------
	 const edm::Ref< L1GctEtMiss >& gctEtMiss() const
	 { return m_gctEtMiss ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 L1EtMissParticle(const L1EtMissParticle&); // stop default

	 const L1EtMissParticle& operator=(const L1EtMissParticle&); // stop default

	 // ---------- member data --------------------------------
	 edm::RefProd< L1GctEtMiss > m_gctEtMiss ;
   };
}

#endif
