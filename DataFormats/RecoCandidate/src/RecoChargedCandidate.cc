// $Id: RecoChargedCandidate.cc,v 1.2 2006/04/26 07:56:21 llista Exp $
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
  const RecoCandidate * dstc = dynamic_cast<const RecoCandidate *>( & c );
  if ( dstc == 0 ) return false;
  if ( checkOverlap( track(), dstc->track() ) ) return true;
  return false;
}
