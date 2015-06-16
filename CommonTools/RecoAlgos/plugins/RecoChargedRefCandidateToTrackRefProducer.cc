#include "CommonTools/RecoAlgos/src/RecoChargedRefCandidateToTrackRef.h"
#include "CommonTools/RecoAlgos/src/CandidateProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
  edm::View<reco::RecoChargedRefCandidate>,
  reco::TrackRefVector,
  AnySelector,
  converter::RecoChargedRefCandidateToTrackRef
  > RecoChargedRefCandidateToTrackRefProducer;

DEFINE_FWK_MODULE(RecoChargedRefCandidateToTrackRefProducer);
