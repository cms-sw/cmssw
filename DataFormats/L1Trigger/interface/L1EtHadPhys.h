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
// $Id$
//

// system include files

// user include files
#include "DataFormats/Common/interface/RefProd.h"

// forward declarations
class L1GctEtHad ;

namespace level1 {

   class L1EtHadPhys
   {

      public:
	 L1EtHadPhys();
	 virtual ~L1EtHadPhys();

	 // ---------- const member functions ---------------------
         float value() const { return m_value ; }

         const edm::RefProd< L1GctEtHad >& gctEtHad() const
         { return m_gctEtHad ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 L1EtHadPhys(const L1EtHadPhys&); // stop default

	 const L1EtHadPhys& operator=(const L1EtHadPhys&); // stop default

	 // ---------- member data --------------------------------
	 float m_value ;
         edm::RefProd< L1GctEtHad > m_gctEtHad ;
   };
}

#endif
