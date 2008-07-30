// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtMissParticle
// 
/**\class L1EtMissParticle \file L1EtMissParticle.cc DataFormats/L1Trigger/src/L1EtMissParticle.cc \author Werner Sun
*/
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 18:22:52 EDT 2006
// $Id: L1EtMissParticle.cc,v 1.5 2007/10/01 19:34:57 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

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
L1EtMissParticle::L1EtMissParticle()
{
}

L1EtMissParticle::L1EtMissParticle(
   const LorentzVector& p4,
   const double& etTotal,
   const double& etHad,
   const edm::RefProd< L1GctEtMiss >& aEtMissRef,
   const edm::RefProd< L1GctEtTotal >& aEtTotalRef,
   const edm::RefProd< L1GctEtHad >& aEtHadRef )
   : LeafCandidate( ( char ) 0, p4 ),
     etTot_( etTotal ),
     etHad_( etHad ),
     etMissRef_( aEtMissRef ),
     etTotRef_( aEtTotalRef ),
     etHadRef_( aEtHadRef )
{
}

L1EtMissParticle::L1EtMissParticle(
   const PolarLorentzVector& p4,
   const double& etTotal,
   const double& etHad,
   const edm::RefProd< L1GctEtMiss >& aEtMissRef,
   const edm::RefProd< L1GctEtTotal >& aEtTotalRef,
   const edm::RefProd< L1GctEtHad >& aEtHadRef )
   : LeafCandidate( ( char ) 0, p4 ),
     etTot_( etTotal ),
     etHad_( etHad ),
     etMissRef_( aEtMissRef ),
     etTotRef_( aEtTotalRef ),
     etHadRef_( aEtHadRef )
{
}

// L1EtMissParticle::L1EtMissParticle(const L1EtMissParticle& rhs)
// {
//    // do actual copying here;
// }

// L1EtMissParticle::~L1EtMissParticle()
// {
// }

//
// assignment operators
//
// const L1EtMissParticle& L1EtMissParticle::operator=(const L1EtMissParticle& rhs)
// {
//   //An exception safe implementation is
//   L1EtMissParticle temp(rhs);
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
