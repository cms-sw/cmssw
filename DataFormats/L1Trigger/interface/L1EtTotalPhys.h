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
// $Id$
//

// system include files

// user include files
#include "DataFormats/Common/interface/RefProd.h"

// forward declarations
class L1GctEtTotal ;

namespace level1 {

   class L1EtTotalPhys
   {

      public:
	 L1EtTotalPhys();
	 virtual ~L1EtTotalPhys();

	 // ---------- const member functions ---------------------
         float value() const { return m_value ; }

         const edm::RefProd< L1GctEtTotal >& gctEtTotal() const
         { return m_gctEtTotal ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 L1EtTotalPhys(const L1EtTotalPhys&); // stop default

	 const L1EtTotalPhys& operator=(const L1EtTotalPhys&); // stop default

	 // ---------- member data --------------------------------
	 float m_value ;
         edm::RefProd< L1GctEtTotal > m_gctEtTotal ;
   };
}

#endif
