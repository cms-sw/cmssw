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
// $Id: L1EtMissParticle.cc,v 1.8 2009/03/20 15:51:07 wsun Exp $
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
	EtMissType type,
	const double& etTotal,
	const edm::Ref< L1GctEtMissCollection >& aEtMissRef,
	const edm::Ref< L1GctEtTotalCollection >& aEtTotalRef,
	const edm::Ref< L1GctHtMissCollection >& aHtMissRef,
	const edm::Ref< L1GctEtHadCollection >& aEtHadRef,
	int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     type_( type ),
     etTot_( etTotal ),
     etMissRef_( aEtMissRef ),
     etTotRef_( aEtTotalRef ),
     htMissRef_( aHtMissRef ),
     etHadRef_( aEtHadRef ),
     bx_( bx )
{
}


L1EtMissParticle::L1EtMissParticle(
	const PolarLorentzVector& p4,
	EtMissType type,
	const double& etTotal,
	const edm::Ref< L1GctEtMissCollection >& aEtMissRef,
	const edm::Ref< L1GctEtTotalCollection >& aEtTotalRef,
	const edm::Ref< L1GctHtMissCollection >& aHtMissRef,
	const edm::Ref< L1GctEtHadCollection >& aEtHadRef,
	int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     type_( type ),
     etTot_( etTotal ),
     etMissRef_( aEtMissRef ),
     etTotRef_( aEtTotalRef ),
     htMissRef_( aHtMissRef ),
     etHadRef_( aEtHadRef ),
     bx_( bx )
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
