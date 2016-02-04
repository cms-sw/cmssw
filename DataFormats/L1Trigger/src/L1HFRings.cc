// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1HFRings
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Fri Mar 20 12:16:54 CET 2009
// $Id: L1HFRings.cc,v 1.2 2009/03/22 16:11:30 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1HFRings.h"

using namespace l1extra ;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1HFRings::L1HFRings()
{
}

L1HFRings::L1HFRings(
   const double* hfEtSums, // array of etSums
   const int* hfBitCounts, // array of bitCounts
   const edm::Ref< L1GctHFRingEtSumsCollection >& aHFEtSumsRef,
   const edm::Ref< L1GctHFBitCountsCollection >& aHFBitCountsRef,
   int bx )
  : m_etSumsRef( aHFEtSumsRef ),
    m_bitCountsRef( aHFBitCountsRef ),
    m_bx( bx )
{
  for( int i = 0 ; i < kNumRings ; ++i )
    {
      m_ringEtSums[ i ] = hfEtSums[ i ] ;
      m_ringBitCounts[ i ] = hfBitCounts[ i ] ;
    }
}

// L1HFRings::L1HFRings(const L1HFRings& rhs)
// {
//    // do actual copying here;
// }

L1HFRings::~L1HFRings()
{
}

//
// assignment operators
//
// const L1HFRings& L1HFRings::operator=(const L1HFRings& rhs)
// {
//   //An exception safe implementation is
//   L1HFRings temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
