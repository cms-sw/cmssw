// $Id: RecoEcalCandidate.cc,v 1.1 2006/07/26 07:42:08 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

using namespace reco;

RecoEcalCandidate::~RecoEcalCandidate() { }

RecoEcalCandidate * RecoEcalCandidate::clone() const { 
  return new RecoEcalCandidate( * this ); 
}

SuperClusterRef RecoEcalCandidate::superCluster() const {
  return superCluster_;
}

bool RecoEcalCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   checkOverlap( superCluster(), o->superCluster() ) 
	   );
}
