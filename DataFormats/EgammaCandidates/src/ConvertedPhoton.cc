
#include "DataFormats/EgammaCandidates/interface/ConvertedPhoton.h"

using namespace reco;

ConvertedPhoton::~ConvertedPhoton() { }

ConvertedPhoton * ConvertedPhoton::clone() const { 
  return new ConvertedPhoton( * this ); 
}

reco::SuperClusterRef ConvertedPhoton::superCluster() const {
  return superCluster_;
}




const std::vector<reco::TrackRef>&  ConvertedPhoton::tracks() const { 
   return tracks_;
}


TrackRef ConvertedPhoton::track( size_t ind ) const {
  return tracks_[ind];
}



bool ConvertedPhoton::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( superCluster(), o->superCluster() ) )
	   );
  return false;
}
   
