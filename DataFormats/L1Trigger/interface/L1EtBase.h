#ifndef L1Trigger_L1EtBase_h
#define L1Trigger_L1EtBase_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtBase
// 
/**\class L1EtBase L1EtBase.h DataFormats/L1Trigger/interface/L1EtBase.h

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
#include "DataFormats/L1Trigger/interface/L1PhysObjectBase.h"

// forward declarations

namespace l1extra {

   class L1EtBase : public L1PhysObjectBase
   {

      public:
	 L1EtBase();
	 L1EtBase( const L1Ref& aRef,
		   float aEtValue ) ;

	 virtual ~L1EtBase();

	 // ---------- const member functions ---------------------
	 const float& etValue() const { return etValue_ ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------
	 void setEtValue( const float& aValue ) { etValue_ = aValue ; }

      private:
	 // L1EtBase(const L1EtBase&); // stop default

	 // const L1EtBase& operator=(const L1EtBase&);
	 // stop default

	 // ---------- member data --------------------------------
	 float etValue_ ;
   };

}

#endif
