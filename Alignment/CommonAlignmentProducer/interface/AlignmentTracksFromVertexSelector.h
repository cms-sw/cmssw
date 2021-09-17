#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentTrackFromVertexSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentTrackFromVertexSelector_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include <vector>

namespace edm {
  class Event;
  class ParameterSet;
}  // namespace edm

class TrackingRecHit;

class AlignmentTrackFromVertexSelector {
public:
  typedef std::vector<const reco::Track*> Tracks;

  /// constructor
  AlignmentTrackFromVertexSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  /// destructor
  ~AlignmentTrackFromVertexSelector();

  /// select tracks
  Tracks select(const edm::Handle<reco::TrackCollection>& tc,
                const edm::Event& evt,
                const edm::EventSetup& setup) const;

  // compute closest vertex
  const reco::Vertex* findClosestVertex(const reco::TrackCollection& leptonTracks,
                                        const reco::VertexCollection* vertices,
                                        const edm::EventSetup& setup) const;

private:
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<reco::TrackCollection> diLeptonToken_;
  bool useClosestVertex_;
  unsigned int vertexIndex_;
};

#endif
