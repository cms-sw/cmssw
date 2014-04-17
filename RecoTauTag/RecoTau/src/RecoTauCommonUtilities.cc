#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <algorithm>

typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;
typedef PFCandPtrs::iterator PFCandIter;

namespace reco { namespace tau {

std::vector<PFCandidatePtr>
flattenPiZeros(const std::vector<RecoTauPiZero>::const_iterator& piZerosBegin, const std::vector<RecoTauPiZero>::const_iterator& piZerosEnd) {
  std::vector<PFCandidatePtr> output;

  for(std::vector<RecoTauPiZero>::const_iterator piZero = piZerosBegin;
      piZero != piZerosEnd; ++piZero) {
    for(size_t iDaughter = 0; iDaughter < piZero->numberOfDaughters();
        ++iDaughter) {
      output.push_back(PFCandidatePtr(piZero->daughterPtr(iDaughter)));
    }
  }
  return output;
}

std::vector<PFCandidatePtr>
flattenPiZeros(const std::vector<RecoTauPiZero>& piZeros) {
  return flattenPiZeros(piZeros.begin(), piZeros.end()); 
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
