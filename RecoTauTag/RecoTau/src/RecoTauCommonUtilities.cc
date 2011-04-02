#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <algorithm>

typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;
typedef PFCandPtrs::iterator PFCandIter;

namespace reco { namespace tau {

// Find the vertex in [vertices] that is closest in Z to the lead track of the
// jet.
reco::VertexRef closestVertex(
    const edm::Handle<reco::VertexCollection>& vertices,
    const reco::PFJet& jet) {
  // Take the first one if exists
  reco::VertexRef selectedVertex = vertices->size() ?
    reco::VertexRef(vertices, 0) : reco::VertexRef();
  double minDZ = std::numeric_limits<double>::infinity();
  // Find the lead charged object in the jet (true mean sort)
  std::vector<PFCandidatePtr> tracks = pfChargedCands(jet, true);
  if (!tracks.size())
    return selectedVertex;
  reco::TrackRef leadTrack = tracks[0]->trackRef();
  if (leadTrack.isNull()) {
    edm::LogError("NullTrackRefInLeadPFCharged")
      << "The leading *charged* PF cand in the jet has an invalid TrackRef!"
      << " LeadPFCand: " << *tracks[0];
    return selectedVertex;
  }
  // Loop over all the vertices and check if they are closer in z to the
  // current track.
  //std::cout << "track is @ z = " << leadTrack->vz()
    //<< " with pt: " << leadTrack->pt() << std::endl;
  for (unsigned int ivtx = 0; ivtx < vertices->size(); ivtx++) {
    reco::VertexRef pvCand(vertices, ivtx);
    //std::cout << "checking vertex i @ z = " << pvCand->z() << std::endl;
    double dz = std::abs(leadTrack->dz(pvCand->position()));
    if (dz < minDZ) {
      minDZ = dz;
      selectedVertex = pvCand;
    }
  }
  return selectedVertex;
}

std::vector<PFCandidatePtr>
flattenPiZeros( const std::vector<RecoTauPiZero>& piZeros) {
  std::vector<PFCandidatePtr> output;

  for(std::vector<RecoTauPiZero>::const_iterator piZero = piZeros.begin();
      piZero != piZeros.end(); ++piZero) {
    for(size_t iDaughter = 0; iDaughter < piZero->numberOfDaughters();
        ++iDaughter) {
      output.push_back(PFCandidatePtr(piZero->daughterPtr(iDaughter)));
    }
  }
  return output;
}

std::vector<reco::PFCandidatePtr> pfCandidates(const reco::PFJet& jet,
    int particleId, bool sort) {
  PFCandPtrs pfCands = jet.getPFConstituents();
  PFCandPtrs selectedPFCands = filterPFCandidates(
      pfCands.begin(), pfCands.end(), particleId, sort);
  return selectedPFCands;
}

std::vector<reco::PFCandidatePtr> pfCandidates(const reco::PFJet& jet,
    const std::vector<int>& particleIds, bool sort) {
  PFCandPtrs output;
  // Get each desired candidate type, unsorted for now
  for(std::vector<int>::const_iterator particleId = particleIds.begin();
      particleId != particleIds.end(); ++particleId) {
    PFCandPtrs selectedPFCands = pfCandidates(jet, *particleId, false);
    output.insert(output.end(), selectedPFCands.begin(), selectedPFCands.end());
  }
  if (sort) std::sort(output.begin(), output.end(), SortPFCandsDescendingPt());
  return output;
}

std::vector<reco::PFCandidatePtr> pfGammas(const reco::PFJet& jet, bool sort) {
  return pfCandidates(jet, reco::PFCandidate::gamma, sort);
}

std::vector<reco::PFCandidatePtr> pfChargedCands(const reco::PFJet& jet,
                                                 bool sort) {
  PFCandPtrs output;
  PFCandPtrs mus = pfCandidates(jet, reco::PFCandidate::mu, false);
  PFCandPtrs es = pfCandidates(jet, reco::PFCandidate::e, false);
  PFCandPtrs chs = pfCandidates(jet, reco::PFCandidate::h, false);
  output.reserve(mus.size() + es.size() + chs.size());
  output.insert(output.end(), mus.begin(), mus.end());
  output.insert(output.end(), es.begin(), es.end());
  output.insert(output.end(), chs.begin(), chs.end());
  if (sort) std::sort(output.begin(), output.end(), SortPFCandsDescendingPt());
  return output;
}



} }
