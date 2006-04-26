// $Id: RecoChargedCandidate.cc,v 1.1 2006/02/28 10:59:16 llista Exp $
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
  TrackRef t1 = track(), t2 = dstc->track();
  if ( ! t1.isNull() && ! t2.isNull() && t1 == t2 ) return true;
  return false;
}
