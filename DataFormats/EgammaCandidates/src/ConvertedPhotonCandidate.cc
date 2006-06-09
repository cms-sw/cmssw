
#include "DataFormats/EgammaCandidates/interface/ConvertedPhotonCandidate.h"

using namespace reco;

ConvertedPhotonCandidate::~ConvertedPhotonCandidate() { }

ConvertedPhotonCandidate * ConvertedPhotonCandidate::clone() const { 
  return new ConvertedPhotonCandidate( * this ); 
}

reco::SuperClusterRef ConvertedPhotonCandidate::superCluster() const {
  return superCluster_;
}

bool ConvertedPhotonCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( superCluster(), o->superCluster() ) )
	   );
  return false;
}
