#ifndef L1Trigger_L1EtTotalPhys_h
#define L1Trigger_L1EtTotalPhys_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtTotalPhys
// 
/**\class L1EtTotalPhys L1EtTotalPhys.h DataFormats/L1Trigger/interface/L1EtTotalPhys.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1EtTotalPhys.h,v 1.1 2006/07/17 20:35:19 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EtBase.h"

// forward declarations
class L1GctEtTotal ;

namespace l1extra {

   class L1EtTotalPhys : public L1EtBase
   {

      public:
	 L1EtTotalPhys();
	 L1EtTotalPhys( const L1Ref& aRef,
			float aEtValue ) ;

	 virtual ~L1EtTotalPhys();

	 // ---------- const member functions ---------------------
         const L1GctEtTotal* gctEtTotal() const ;

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1EtTotalPhys(const L1EtTotalPhys&); // stop default

	 // const L1EtTotalPhys& operator=(const L1EtTotalPhys&); // stop default

	 // ---------- member data --------------------------------
   };
}

#endif
