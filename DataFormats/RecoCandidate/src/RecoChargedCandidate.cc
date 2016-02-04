// $Id: RecoChargedCandidate.cc,v 1.6 2007/12/14 12:25:33 llista Exp $
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
