#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentTrackFromVertexCompositeCandidateSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentTrackFromVertexCompositeCandidateSelector_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>

namespace edm {
  class Event;
  class ParameterSet;
}  // namespace edm

class TrackingRecHit;

class AlignmentTrackFromVertexCompositeCandidateSelector {
public:
  typedef std::vector<const reco::Track*> Tracks;

  /// constructor
  AlignmentTrackFromVertexCompositeCandidateSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  /// destructor
  ~AlignmentTrackFromVertexCompositeCandidateSelector();

  /// select tracks
  Tracks select(const edm::Handle<reco::TrackCollection>& tc,
                const edm::Event& evt,
                const edm::EventSetup& setup) const;

private:
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> vccToken_;
};

#endif
-- dummy change --
