#ifndef DataFormats_L1Trigger_L1HFRings_h
#define DataFormats_L1Trigger_L1HFRings_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1HFRings
// 
/**\class L1HFRings L1HFRings.h DataFormats/L1Trigger/interface/L1HFRings.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Mar 14 19:04:20 CET 2009
// $Id: L1HFRings.h,v 1.2 2009/03/22 16:11:30 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/Common/interface/Ref.h"

// forward declarations

namespace l1extra {

class L1HFRings
{

   public:
     enum HFRingLabels { kRing1PosEta, kRing1NegEta, kRing2PosEta, 
			 kRing2NegEta, kNumRings } ;

      L1HFRings();

      // Default Refs are null.
      L1HFRings( const double* hfEtSums, // array of etSums
		 const int* hfBitCounts, // array of bitCounts
		 const edm::Ref< L1GctHFRingEtSumsCollection >& aHFEtSumsRef = 
		 edm::Ref< L1GctHFRingEtSumsCollection >(),
		 const edm::Ref< L1GctHFBitCountsCollection >& aHFBitCountsRef 
		 = edm::Ref< L1GctHFBitCountsCollection >(),
		 int bx = 0 ) ;

      virtual ~L1HFRings();

      // ---------- const member functions ---------------------
      double hfEtSum( HFRingLabels i ) const // in  GeV
	{ return m_ringEtSums[ i ] ; }
      int hfBitCount( HFRingLabels i ) const
	{ return m_ringBitCounts [ i ] ; }

      const edm::Ref< L1GctHFRingEtSumsCollection >& gctHFEtSumsRef() const
	{ return m_etSumsRef ; }
      const edm::Ref< L1GctHFBitCountsCollection >& gctHFBitCountsRef() const
	{ return m_bitCountsRef ; }

      const L1GctHFRingEtSums* gctHFEtSums() const
	{ return m_etSumsRef.get() ; }
      const L1GctHFBitCounts* gctHFBitCounts() const
	{ return m_bitCountsRef.get() ; }

      int bx() const { return m_bx ; }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      // L1HFRings(const L1HFRings&); // stop default

      // const L1HFRings& operator=(const L1HFRings&); // stop default

      // ---------- member data --------------------------------
      double m_ringEtSums[ kNumRings ] ;
      int m_ringBitCounts[ kNumRings ] ;

      edm::Ref< L1GctHFRingEtSumsCollection > m_etSumsRef ;
      edm::Ref< L1GctHFBitCountsCollection > m_bitCountsRef ;

      int m_bx ;
};
}

#endif
