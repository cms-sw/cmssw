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
// $Id: L1EtMissParticle.cc,v 1.1 2006/07/26 00:05:39 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

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

L1EtMissParticle::L1EtMissParticle( const LorentzVector& p4,
				    const double& etTotal,
				    const double& etHad,
				    const L1Ref& aEtMissRef,
				    const L1Ref& aEtTotalRef,
				    const L1Ref& aEtHadRef )
//   : ParticleKinematics( p4 ),
   : L1PhysObjectBase( ( char ) 0, p4, aEtMissRef ),
     etTot_( etTotal ),
     etHad_( etHad ),
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

const L1GctEtMiss*
L1EtMissParticle::gctEtMiss() const
{
   return dynamic_cast< const L1GctEtMiss* >( triggerObjectPtr() ) ;
}

const L1GctEtTotal*
L1EtMissParticle::gctEtTotal() const
{
   return dynamic_cast< const L1GctEtTotal* >( etTotRef_.get() ) ;
}

const L1GctEtHad*
L1EtMissParticle::gctEtHad() const
{
   return dynamic_cast< const L1GctEtHad* >( etHadRef_.get() ) ;
}

//
// const member functions
//

//
// static member functions
//
