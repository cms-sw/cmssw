#ifndef L1Trigger_L1PhysObjectBase_h
#define L1Trigger_L1PhysObjectBase_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1PhysObjectBase
// 
/**\class L1PhysObjectBase L1PhysObjectBase.h DataFormats/L1Trigger/interface/L1PhysObjectBase.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 15:09:21 EDT 2006
// $Id: L1PhysObjectBase.h,v 1.2 2006/07/26 20:41:30 wsun Exp $
// $Log: L1PhysObjectBase.h,v $
// Revision 1.2  2006/07/26 20:41:30  wsun
// Added implementation of L1ParticleMap.
//
// Revision 1.1  2006/07/26 00:05:39  wsun
// Structural mods for HLT use.
//
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/RefToBase.h"

// forward declarations
class L1TriggerObject ;

namespace l1extra {

   class L1PhysObjectBase : public reco::LeafCandidate
   {

      public:
	 typedef edm::RefToBase< L1TriggerObject > L1Ref ;

	 L1PhysObjectBase();
	 L1PhysObjectBase( Charge q,
			   const LorentzVector& p4,
			   const L1Ref& aRef ) ;

	 virtual ~L1PhysObjectBase();

	 // ---------- const member functions ---------------------
         const L1Ref& triggerObjectRef() const
         { return ref_ ; }

	 const L1TriggerObject* triggerObjectPtr() const
	 { return ref_.get() ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1PhysObjectBase(const L1PhysObjectBase&); // stop default

	 // const L1PhysObjectBase& operator=(const L1PhysObjectBase&);
	 // stop default

	 // ---------- member data --------------------------------
	 L1Ref ref_ ;
   };

}

#endif
