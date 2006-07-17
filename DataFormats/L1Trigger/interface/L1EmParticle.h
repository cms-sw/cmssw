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
// $Id$
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/ParticleKinematics.h"
#include "DataFormats/Common/interface/Ref.h"

// forward declarations
class L1GctEmCandCollection ;

namespace level1 {

   class L1EmParticle : public ParticleKinematics
   {

      public:
	 L1EmParticle();
	 virtual ~L1EmParticle();

	 // ---------- const member functions ---------------------
	 const edm::Ref< L1GctEmCand >& gctEmCand() const
	 { return m_gctEmCand ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 L1EmParticle(const L1EmParticle&); // stop default

	 const L1EmParticle& operator=(const L1EmParticle&); // stop default

	 // ---------- member data --------------------------------
	 edm::Ref< L1GctEmCandCollection > m_gctEmCand ;
   };
}

#endif
