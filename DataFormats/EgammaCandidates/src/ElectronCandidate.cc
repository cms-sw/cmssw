// $Id: ElectronCandidate.cc,v 1.4 2006/05/02 10:28:00 llista Exp $
#include "DataFormats/EgammaCandidates/interface/ElectronCandidate.h"

using namespace reco;

ElectronCandidate::~ElectronCandidate() { }

ElectronCandidate * ElectronCandidate::clone() const { 
  return new ElectronCandidate( * this ); 
}

TrackRef ElectronCandidate::track() const {
  return track_;
}

SuperClusterRef ElectronCandidate::superCluster() const {
  return superCluster_;
}

bool ElectronCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( track(), o->track() ) ||
	     checkOverlap( superCluster(), o->superCluster() ) ) 
	   );
  return false;
}
