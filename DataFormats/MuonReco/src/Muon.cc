#include "DataFormats/MuonReco/interface/Muon.h"
using namespace reco;

Muon::Muon(  Charge q, const LorentzVector & p4, const Point & vtx ) : 
      RecoCandidate( q, p4, vtx ) { 
}


bool Muon::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  if ( o == 0 ) return false;
  if ( checkOverlap( track(), o->track() ) ) return true;
  if ( checkOverlap( standAloneMuon(), o->standAloneMuon() ) ) return true;
  if ( checkOverlap( combinedMuon(), o->combinedMuon() ) ) return true;
  if ( checkOverlap( superCluster(), o->superCluster() ) ) return true;
  return false;
}



