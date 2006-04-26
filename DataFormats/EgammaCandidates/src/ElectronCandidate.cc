// $Id: ElectronCandidate.cc,v 1.1 2006/04/21 06:28:47 llista Exp $
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
  const RecoCandidate * dstc = dynamic_cast<const RecoCandidate *>( & c );
  if ( dstc == 0 ) return false;
  TrackRef t1 = track(), t2 = dstc->track();
  if ( ! t1.isNull() && ! t2.isNull() && t1 == t2 ) return true;
  SuperClusterRef s1 = superCluster(), s2 = dstc->superCluster();
  return false;
}
