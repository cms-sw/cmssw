#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"

#include <algorithm>

typedef std::vector<reco::CandidatePtr> CandPtrs;
typedef CandPtrs::iterator CandIter;

namespace reco {
  namespace tau {

    namespace {
      // Re-implemented from PFCandidate.cc
      int translateTypeToAbsPdgId(int type) {
        switch (type) {
          case reco::PFCandidate::h:
            return 211;  // pi+
          case reco::PFCandidate::e:
            return 11;
          case reco::PFCandidate::mu:
            return 13;
          case reco::PFCandidate::gamma:
            return 22;
          case reco::PFCandidate::h0:
            return 130;  // K_L0
          case reco::PFCandidate::h_HF:
            return 1;  // dummy pdg code
          case reco::PFCandidate::egamma_HF:
            return 2;  // dummy pdg code
          case reco::PFCandidate::X:
          default:
            return 0;
        }
      }
    }  // namespace
    std::vector<CandidatePtr> flattenPiZeros(const std::vector<RecoTauPiZero>::const_iterator& piZerosBegin,
                                             const std::vector<RecoTauPiZero>::const_iterator& piZerosEnd) {
      std::vector<CandidatePtr> output;

      for (std::vector<RecoTauPiZero>::const_iterator piZero = piZerosBegin; piZero != piZerosEnd; ++piZero) {
        for (size_t iDaughter = 0; iDaughter < piZero->numberOfDaughters(); ++iDaughter) {
          output.push_back(CandidatePtr(piZero->daughterPtr(iDaughter)));
        }
      }
      return output;
    }

    std::vector<CandidatePtr> flattenPiZeros(const std::vector<RecoTauPiZero>& piZeros) {
      return flattenPiZeros(piZeros.begin(), piZeros.end());
    }

    std::vector<reco::CandidatePtr> pfCandidates(const reco::Jet& jet, int particleId, bool sort) {
      return pfCandidatesByPdgId(jet, translateTypeToAbsPdgId(particleId), sort);
    }

    std::vector<CandidatePtr> pfCandidates(const Jet& jet, const std::vector<int>& particleIds, bool sort) {
      std::vector<int> pdgIds;
      pdgIds.reserve(particleIds.size());
      for (auto particleId : particleIds)
        pdgIds.push_back(translateTypeToAbsPdgId(particleId));
      return pfCandidatesByPdgId(jet, pdgIds, sort);
    }

    std::vector<reco::CandidatePtr> pfCandidatesByPdgId(const reco::Jet& jet, int pdgId, bool sort) {
      CandPtrs pfCands = jet.daughterPtrVector();
      CandPtrs selectedPFCands = filterPFCandidates(pfCands.begin(), pfCands.end(), pdgId, sort);
      return selectedPFCands;
    }

    std::vector<reco::CandidatePtr> pfCandidatesByPdgId(const reco::Jet& jet,
                                                        const std::vector<int>& pdgIds,
                                                        bool sort) {
      const CandPtrs& pfCands = jet.daughterPtrVector();
      CandPtrs output;
      // Get each desired candidate type, unsorted for now
      for (std::vector<int>::const_iterator pdgId = pdgIds.begin(); pdgId != pdgIds.end(); ++pdgId) {
        CandPtrs&& selectedPFCands = filterPFCandidates(pfCands.begin(), pfCands.end(), *pdgId, false);
        output.insert(output.end(), selectedPFCands.begin(), selectedPFCands.end());
      }
      if (sort)
        std::sort(output.begin(), output.end(), SortPFCandsDescendingPt());
      return output;
    }

    std::vector<reco::CandidatePtr> pfGammas(const reco::Jet& jet, bool sort) { return pfCandidates(jet, 22, sort); }

    std::vector<reco::CandidatePtr> pfChargedCands(const reco::Jet& jet, bool sort) {
      const CandPtrs& pfCands = jet.daughterPtrVector();
      CandPtrs output;
      CandPtrs&& mus = filterPFCandidates(pfCands.begin(), pfCands.end(), 13, false);
      CandPtrs&& es = filterPFCandidates(pfCands.begin(), pfCands.end(), 11, false);
      CandPtrs&& chs = filterPFCandidates(pfCands.begin(), pfCands.end(), 211, false);
      output.reserve(mus.size() + es.size() + chs.size());
      output.insert(output.end(), mus.begin(), mus.end());
      output.insert(output.end(), es.begin(), es.end());
      output.insert(output.end(), chs.begin(), chs.end());
      if (sort)
        std::sort(output.begin(), output.end(), SortPFCandsDescendingPt());
      return output;
    }

    math::XYZPointF atECALEntrance(const reco::Candidate* part, double bField) {
      const reco::PFCandidate* pfCand = dynamic_cast<const reco::PFCandidate*>(part);
      if (pfCand)
        return pfCand->positionAtECALEntrance();

      math::XYZPointF pos;
      BaseParticlePropagator theParticle = BaseParticlePropagator(
          RawParticle(math::XYZTLorentzVector(part->px(), part->py(), part->pz(), part->energy()),
                      math::XYZTLorentzVector(part->vertex().x(), part->vertex().y(), part->vertex().z(), 0.)),
          part->charge(),
          0.,
          0.,
          bField);
      theParticle.propagateToEcalEntrance(false);
      if (theParticle.getSuccess() != 0) {
        pos = math::XYZPointF(theParticle.particle().vertex());
      }
      return pos;
    }

  }  // namespace tau
}  // namespace reco
