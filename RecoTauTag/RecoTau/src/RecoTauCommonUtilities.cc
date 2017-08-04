#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/ParticleFlowCandidate/interface/CandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <algorithm>

typedef std::vector<reco::CandidatePtr> CandPtrs;
typedef CandPtrs::iterator CandIter;

namespace reco { namespace tau {

std::vector<CandidatePtr>
flattenPiZeros(const std::vector<RecoTauPiZero>::const_iterator& piZerosBegin, const std::vector<RecoTauPiZero>::const_iterator& piZerosEnd) {
  std::vector<CandidatePtr> output;

  for(std::vector<RecoTauPiZero>::const_iterator piZero = piZerosBegin;
      piZero != piZerosEnd; ++piZero) {
    for(size_t iDaughter = 0; iDaughter < piZero->numberOfDaughters();
        ++iDaughter) {
      output.push_back(CandidatePtr(piZero->daughterPtr(iDaughter)));
    }
  }
  return output;
}

std::vector<CandidatePtr>
flattenPiZeros(const std::vector<RecoTauPiZero>& piZeros) {
  return flattenPiZeros(piZeros.begin(), piZeros.end()); 
}

std::vector<reco::CandidatePtr> pfCandidates(const reco::Jet& jet,
    int pdgId, bool sort) {
  CandPtrs pfCands = jet.getPFConstituents();
  CandPtrs selectedPFCands = filterPFCandidates(
      pfCands.begin(), pfCands.end(), pdgId, sort);
  return selectedPFCands;
}

std::vector<reco::CandidatePtr> pfCandidates(const reco::Jet& jet,
    const std::vector<int>& pdgIds, bool sort) {
  CandPtrs&& pfCands = jet.getPFConstituents();
  CandPtrs output;
  // Get each desired candidate type, unsorted for now
  for(std::vector<int>::const_iterator pdgId = pdgIds.begin();
      pdgId != pdgIds.end(); ++pdgId) {
    CandPtrs&& selectedPFCands = filterPFCandidates(pfCands.begin(), pfCands.end(), *pdgId, false);
    output.insert(output.end(), selectedPFCands.begin(), selectedPFCands.end());
  }
  if (sort) std::sort(output.begin(), output.end(), SortPFCandsDescendingPt());
  return output;
}

std::vector<reco::CandidatePtr> pfGammas(const reco::Jet& jet, bool sort) {
  return pfCandidates(jet, 22, sort);
}

std::vector<reco::CandidatePtr> pfChargedCands(const reco::Jet& jet,
                                                 bool sort) {
  CandPtrs&& pfCands = jet.getPFConstituents();
  CandPtrs output;
  CandPtrs&& mus = filterPFCandidates(pfCands.begin(), pfCands.end(), 13, false);
  CandPtrs&& es = filterPFCandidates(pfCands.begin(), pfCands.end(), 11, false);
  CandPtrs&& chs = filterPFCandidates(pfCands.begin(), pfCands.end(), 211, false);
  output.reserve(mus.size() + es.size() + chs.size());
  output.insert(output.end(), mus.begin(), mus.end());
  output.insert(output.end(), es.begin(), es.end());
  output.insert(output.end(), chs.begin(), chs.end());
  if (sort) std::sort(output.begin(), output.end(), SortPFCandsDescendingPt());
  return output;
}

math::XYZPoint atECALEntrance(const reco::Candidate* part) {
  const reco::PFCandidate* pfCand = dynamic_cast<const reco::PFCandidate>(part);
  if (pfCand)
    return pfCand->positionAtECALEntrance();

  math::XYZPoint pos;
  BaseParticlePropagator theParticle =
    BaseParticlePropagator(RawParticle(math::XYZTLorentzVector(part->px(),
                     part->py(),
                     part->pz(),
                     part->energy()),
               math::XYZTLorentzVector(part->vertex().x(),
                     part->vertex().y(),
                     part->vertex().z(),
                     0.)), 
         0.,0.,bField_);
  theParticle.setCharge(part->charge());
  theParticle.propagateToEcalEntrance(false);
  if(theParticle.getSuccess()!=0){
    pos = math::XYZPoint(theParticle.vertex());
  }
  return pos;
}


} }
