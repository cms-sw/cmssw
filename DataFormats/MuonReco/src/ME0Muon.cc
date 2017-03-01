/** \file ME0Muon.cc
 *
 *  \author David Nash
 */

#include "DataFormats/MuonReco/interface/ME0Muon.h"
using namespace reco;

ME0Muon::ME0Muon() {
}

bool ME0Muon::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( track(), o->track() ))
	   );
}
