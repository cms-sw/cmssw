// -*- C++ -*-
//
// Package:     EgammaCandidates
// Class  :     SiStripElectronCandidate
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 15:48:28 EDT 2006
// $Id$
//

// system include files

// user include files
#include "DataFormats/EgammaCandidates/interface/SiStripElectronCandidate.h"

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

// SiStripElectronCandidate::SiStripElectronCandidate(const SiStripElectronCandidate& rhs)
// {
//    // do actual copying here;
// }

SiStripElectronCandidate::~SiStripElectronCandidate() { }

//
// assignment operators
//
// const SiStripElectronCandidate& SiStripElectronCandidate::operator=(const SiStripElectronCandidate& rhs)
// {
//   //An exception safe implementation is
//   SiStripElectronCandidate temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

SiStripElectronCandidate * SiStripElectronCandidate::clone() const { 
  return new SiStripElectronCandidate( * this ); 
}

//
// member functions
//

TrackRef SiStripElectronCandidate::track() const {
  return track_;
}

SuperClusterRef SiStripElectronCandidate::superCluster() const {
  return superCluster_;
}

bool SiStripElectronCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && ! 
	   ( checkOverlap( track(), o->track() ) ||
	     checkOverlap( superCluster(), o->superCluster() ) ) 
	   );
  return false;
}

//
// const member functions
//

//
// static member functions
//
