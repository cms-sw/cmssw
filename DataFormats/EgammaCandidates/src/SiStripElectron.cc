// -*- C++ -*-
//
// Package:     EgammaCandidates
// Class  :     SiStripElectron
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 15:48:28 EDT 2006
// $Id: SiStripElectron.cc,v 1.3 2008/04/21 14:05:24 llista Exp $
//

// system include files

// user include files
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 

using namespace reco;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

// SiStripElectron::SiStripElectron(const SiStripElectron& rhs)
// {
//    // do actual copying here;
// }

SiStripElectron::~SiStripElectron() { }

//
// assignment operators
//
// const SiStripElectron& SiStripElectron::operator=(const SiStripElectron& rhs)
// {
//   //An exception safe implementation is
//   SiStripElectron temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

SiStripElectron * SiStripElectron::clone() const { 
  return new SiStripElectron( * this ); 
}

//
// member functions
//

SuperClusterRef SiStripElectron::superCluster() const {
  return superCluster_;
}

bool SiStripElectron::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && ! 
	   ( checkOverlap( track(), o->track() ) ||
	     checkOverlap( superCluster(), o->superCluster() ) ) 
	   );
  return false;
}

bool SiStripElectron::isElectron() const {
  return true;
}


//
// const member functions
//

//
// static member functions
//
