// $Id: RecoChargedCandidate.cc,v 1.3 2006/02/21 10:37:36 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

using namespace reco;

RecoChargedCandidate::~RecoChargedCandidate() { }

RecoChargedCandidate * RecoChargedCandidate::clone() const { 
  return new RecoChargedCandidate( * this ); 
}

TrackRef RecoChargedCandidate::track() const {
  return track_;
}
