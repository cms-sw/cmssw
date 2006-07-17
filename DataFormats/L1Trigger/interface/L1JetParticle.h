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
// $Id$
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/ParticleKinematics.h"
#include "DataFormats/Common/interface/Ref.h"

// forward declarations
class L1GctJetCandCollection ;

namespace level1 {

   class L1JetParticle : public ParticleKinematics
   {

      public:
	 L1JetParticle();
	 virtual ~L1JetParticle();

	 // ---------- const member functions ---------------------
	 const edm::Ref< L1GctJetCand >& gctJetCand() const
	 { return m_gctJetCand ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 L1JetParticle(const L1JetParticle&); // stop default

	 const L1JetParticle& operator=(const L1JetParticle&); // stop default

	 // ---------- member data --------------------------------
	 edm::Ref< L1GctJetCandCollection > m_gctJetCand ;
   };
}

#endif
