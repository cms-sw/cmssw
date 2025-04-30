#include "Alignment/CommonAlignmentProducer/interface/AlignmentTracksFromVertexCompositeCandidateSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackReco/interface/Track.h"

// constructor ----------------------------------------------------------------
AlignmentTrackFromVertexCompositeCandidateSelector::AlignmentTrackFromVertexCompositeCandidateSelector(
    const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : vccToken_(iC.consumes<reco::VertexCompositeCandidateCollection>(
          cfg.getParameter<edm::InputTag>("vertexCompositeCandidates"))) {}

// destructor -----------------------------------------------------------------
AlignmentTrackFromVertexCompositeCandidateSelector::~AlignmentTrackFromVertexCompositeCandidateSelector() {}

// do selection ---------------------------------------------------------------
AlignmentTrackFromVertexCompositeCandidateSelector::Tracks AlignmentTrackFromVertexCompositeCandidateSelector::select(
    const edm::Handle<reco::TrackCollection>& tc, const edm::Event& evt, const edm::EventSetup& setup) const {
  Tracks result;

  std::vector<unsigned int> theV0keys;

  edm::Handle<reco::VertexCompositeCandidateCollection> vccHandle;
  evt.getByToken(vccToken_, vccHandle);

  if (vccHandle.isValid()) {
    // Loop over VertexCompositeCandidates and associate tracks
    for (const auto& vcc : *vccHandle) {
      for (size_t i = 0; i < vcc.numberOfDaughters(); ++i) {
        LogDebug("AlignmentTrackFromVertexCompositeCandidateSelector") << "daughter: " << i << std::endl;
        const reco::Candidate* daughter = vcc.daughter(i);
        const reco::RecoChargedCandidate* chargedDaughter = dynamic_cast<const reco::RecoChargedCandidate*>(daughter);
        if (chargedDaughter) {
          LogDebug("AlignmentTrackFromVertexCompositeCandidateSelector") << "charged daughter: " << i << std::endl;
          const reco::TrackRef trackRef = chargedDaughter->track();
          if (trackRef.isNonnull()) {
            LogDebug("AlignmentTrackFromVertexCompositeCandidateSelector")
                << "charged daughter has non-null trackref: " << i << std::endl;
            theV0keys.push_back(trackRef.key());
          }
        }
      }
    }
  } else {
    edm::LogError("AlignmentTrackFromVertexCompositeCandidateSelector")
        << "Error >> Failed to get VertexCompositeCandidateCollection";
  }

  LogDebug("AlignmentTrackFromVertexCompositeCandidateSelector")
      << "collection will have size: " << theV0keys.size() << std::endl;

  if (tc.isValid()) {
    int indx(0);
    // put the track in the collection is it was used for the vertex
    for (reco::TrackCollection::const_iterator tk = tc->begin(); tk != tc->end(); ++tk, ++indx) {
      reco::TrackRef trackRef = reco::TrackRef(tc, indx);
      if (std::find(theV0keys.begin(), theV0keys.end(), trackRef.key()) != theV0keys.end()) {
        LogDebug("AlignmentTrackFromVertexSelector") << "track index: " << indx << "filling result vector" << std::endl;
        result.push_back(&(*tk));
      }  // if a valid key is found
    }  // end loop over tracks
  }  // if the handle is valid

  LogDebug("AlignmentTrackFromVertexCompositeCandidateSelector")
      << "collection will have size: " << result.size() << std::endl;

  return result;
}
-- dummy change --
