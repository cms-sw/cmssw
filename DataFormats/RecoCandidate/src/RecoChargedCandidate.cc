// $Id: RecoChargedCandidate.cc,v 1.5 2006/05/31 12:45:46 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

using namespace reco;

RecoChargedCandidate::~RecoChargedCandidate() { }

RecoChargedCandidate * RecoChargedCandidate::clone() const { 
  return new RecoChargedCandidate( * this ); 
}

TrackRef RecoChargedCandidate::track() const {
  return track_;
}

bool RecoChargedCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return (o != 0 && 
	  (checkOverlap(track(), o->track()) || 
	   checkOverlap(track(), o->standAloneMuon()) ||
	   checkOverlap(track(), o->combinedMuon()))
	  );
}
