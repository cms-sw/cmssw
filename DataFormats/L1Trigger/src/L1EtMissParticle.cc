// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtMissParticle
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Tue Jul 25 18:22:52 EDT 2006
// $Id: L1EtMissParticle.cc,v 1.2 2006/08/02 14:22:33 wsun Exp $
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
//   : ParticleKinematics( p4 ),
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

L1EtMissParticle::~L1EtMissParticle()
{
}

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
