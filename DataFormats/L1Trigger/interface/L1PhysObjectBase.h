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
// $Id$
// $Log$
//

// system include files

// user include files
#include "DataFormats/Common/interface/RefToBase.h"

// forward declarations
class L1TriggerObject ;

namespace l1extra {

   class L1PhysObjectBase
   {

      public:
	 enum L1PhysObjectType {
	    kEM,
	    kJet,
	    kMuon,
	    kEtMiss,
	    kEtSum,
	    kNumberPhysObjectTypes
	 } ;

	 typedef edm::RefToBase< L1TriggerObject > L1Ref ;

	 L1PhysObjectBase();
	 L1PhysObjectBase( const L1Ref& aRef,
			   L1PhysObjectType aType ) ;

	 virtual ~L1PhysObjectBase();

	 // ---------- const member functions ---------------------
         const L1Ref& triggerObjectRef() const
         { return ref_ ; }

	 const L1TriggerObject* triggerObjectPtr() const
	 { return ref_.get() ; }

	 L1PhysObjectType objectType() const
	 { return objectType_ ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1PhysObjectBase(const L1PhysObjectBase&); // stop default

	 // const L1PhysObjectBase& operator=(const L1PhysObjectBase&);
	 // stop default

	 // ---------- member data --------------------------------
	 L1Ref ref_ ;
	 L1PhysObjectType objectType_ ;
   };

}

#endif
