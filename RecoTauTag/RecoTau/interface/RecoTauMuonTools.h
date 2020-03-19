#ifndef RecoTauTag_RecoTau_RecoTauMuonTools_h
#define RecoTauTag_RecoTau_RecoTauMuonTools_h

/*
 * RecoTauMuonTools - utilities for muon->tau discrimination.

 */

#include <vector>

#include "DataFormats/PatCandidates/interface/Muon.h"

namespace reco {
  namespace tau {
    void countHits(const reco::Muon& muon,
                   std::vector<int>& numHitsDT,
                   std::vector<int>& numHitsCSC,
                   std::vector<int>& numHitsRPC);
    void countMatches(const reco::Muon& muon,
                      std::vector<int>& numMatchesDT,
                      std::vector<int>& numMatchesCSC,
                      std::vector<int>& numMatchesRPC);
    std::string format_vint(const std::vector<int>& vi);
  }  // namespace tau
}  // namespace reco

#endif
