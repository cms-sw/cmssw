#include "DataFormats/MuonReco/interface/Muon.h"
using namespace reco;

Muon::Muon(  Charge q, const LorentzVector & p4, const Point & vtx ) : 
  RecoCandidate( q, p4, vtx /*, -13 * q */ ) { 
}


bool Muon::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( track(), o->track() ) ||
	     checkOverlap( standAloneMuon(), o->standAloneMuon() ) ||
	     checkOverlap( combinedMuon(), o->combinedMuon() ) ||
	     checkOverlap( superCluster(), o->superCluster() ) ) 
	   );
}

Muon * Muon::clone() const {
  return new Muon( * this );
}

