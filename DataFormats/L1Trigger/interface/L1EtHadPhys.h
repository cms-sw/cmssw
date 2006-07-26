#ifndef L1Trigger_L1EtHadPhys_h
#define L1Trigger_L1EtHadPhys_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtHadPhys
// 
/**\class L1EtHadPhys L1EtHadPhys.h DataFormats/L1Trigger/interface/L1EtHadPhys.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1EtHadPhys.h,v 1.1 2006/07/17 20:35:19 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EtBase.h"

// forward declarations
class L1GctEtHad ;

namespace l1extra {

   class L1EtHadPhys : public L1EtBase
   {

      public:
	 L1EtHadPhys();
	 L1EtHadPhys( const L1Ref& aRef,
		      float aEtValue ) ;

	 virtual ~L1EtHadPhys();

	 // ---------- const member functions ---------------------
         const L1GctEtHad* gctEtHad() const ;

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1EtHadPhys(const L1EtHadPhys&); // stop default

	 // const L1EtHadPhys& operator=(const L1EtHadPhys&); // stop default

	 // ---------- member data --------------------------------
   };
}

#endif
