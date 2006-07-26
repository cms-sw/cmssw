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
// $Id$
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
				    const L1Ref& aRef )
   : ParticleKinematics( p4 ),
     L1PhysObjectBase( aRef, kEtMiss )
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

//
// const member functions
//

//
// static member functions
//
