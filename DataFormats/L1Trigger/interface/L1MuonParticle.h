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
// $Id: L1MuonParticle.h,v 1.3 2006/08/02 14:22:33 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1PhysObjectBase.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

// forward declarations
class L1MuGMTCand ;

namespace l1extra {

   class L1MuonParticle : public L1PhysObjectBase
   {

      public:
	 L1MuonParticle();

	 L1MuonParticle( Charge q,
			 const LorentzVector& p4,
			 const L1Ref& aRef ) ;

         // Creates null Ref.
         L1MuonParticle( Charge q,
                         const LorentzVector& p4,
                         bool isolated = false,
                         bool mip = false ) ;

	 virtual ~L1MuonParticle();

	 // ---------- const member functions ---------------------
	 const L1MuGMTCand* gmtMuonCand() const ;

         bool isIsolated() const
         { return isolated_ ; }

         bool isMip() const
         { return mip_ ; }

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
   };
}

#endif
