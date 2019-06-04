#ifndef DataFormats_ParticleFlowReco_PFV0_h
#define DataFormats_ParticleFlowReco_PFV0_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"
#include <iostream>
#include <vector>

class Conversion;

namespace reco {

  class PFV0 {
  public:
    /// Default constructor
    PFV0() {}

    PFV0(const reco::VertexCompositeCandidateRef V0,
         const std::vector<reco::PFRecTrackRef>& pftr,
         const std::vector<reco::TrackRef>& tr)
        : originalV0_(V0), pfTracks_(pftr), tracks_(tr) {}

    /// destructor
    ~PFV0(){};

    /// Ref to the original V0
    const reco::VertexCompositeCandidateRef& originalV0() const { return originalV0_; }

    /// Vector of a Refs of PFRecTrack
    const std::vector<reco::PFRecTrackRef>& pfTracks() const { return pfTracks_; }

    /// Vector of a Refs of Track
    const std::vector<reco::TrackRef>& Tracks() const { return tracks_; }

  private:
    reco::VertexCompositeCandidateRef originalV0_;
    std::vector<reco::PFRecTrackRef> pfTracks_;
    std::vector<reco::TrackRef> tracks_;
  };

}  // namespace reco

#endif
