// $Id: RecoChargedCandidate.cc,v 1.5 2006/05/31 12:45:46 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCaloJetCandidate.h"

using namespace reco;

RecoCaloJetCandidate::~RecoCaloJetCandidate() { }

RecoCaloJetCandidate * RecoCaloJetCandidate::clone() const { 
  return new RecoCaloJetCandidate( * this ); 
}

CaloJetRef RecoCaloJetCandidate::caloJet() const {
  return caloJet_;
}

bool RecoCaloJetCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   checkOverlap( caloJet(), o->caloJet() ) 
	   );
}
