#include "RecoTauTag/RecoTau/interface/RecoTauMuonTools.h"

#include "DataFormats/TrackReco/interface/HitPattern.h"

namespace reco {
  namespace tau {
    void countHits(const reco::Muon& muon,
                   std::vector<int>& numHitsDT,
                   std::vector<int>& numHitsCSC,
                   std::vector<int>& numHitsRPC) {
      if (muon.outerTrack().isNonnull()) {
        const reco::HitPattern& muonHitPattern = muon.outerTrack()->hitPattern();
        for (int iHit = 0; iHit < muonHitPattern.numberOfAllHits(reco::HitPattern::TRACK_HITS); ++iHit) {
          uint32_t hit = muonHitPattern.getHitPattern(reco::HitPattern::TRACK_HITS, iHit);
          if (hit == 0)
            break;
          if (muonHitPattern.muonHitFilter(hit) && (muonHitPattern.getHitType(hit) == TrackingRecHit::valid ||
                                                    muonHitPattern.getHitType(hit) == TrackingRecHit::bad)) {
            int muonStation = muonHitPattern.getMuonStation(hit) - 1;  // CV: map into range 0..3
            if (muonStation >= 0 && muonStation < 4) {
              if (muonHitPattern.muonDTHitFilter(hit))
                ++numHitsDT[muonStation];
              else if (muonHitPattern.muonCSCHitFilter(hit))
                ++numHitsCSC[muonStation];
              else if (muonHitPattern.muonRPCHitFilter(hit))
                ++numHitsRPC[muonStation];
            }
          }
        }
      }
    }

    std::string format_vint(const std::vector<int>& vi) {
      std::ostringstream os;
      os << "{ ";
      unsigned numEntries = vi.size();
      for (unsigned iEntry = 0; iEntry < numEntries; ++iEntry) {
        os << vi[iEntry];
        if (iEntry < (numEntries - 1))
          os << ", ";
      }
      os << " }";
      return os.str();
    }

    void countMatches(const reco::Muon& muon,
                      std::vector<int>& numMatchesDT,
                      std::vector<int>& numMatchesCSC,
                      std::vector<int>& numMatchesRPC) {
      const std::vector<reco::MuonChamberMatch>& muonSegments = muon.matches();
      for (std::vector<reco::MuonChamberMatch>::const_iterator muonSegment = muonSegments.begin();
           muonSegment != muonSegments.end();
           ++muonSegment) {
        if (muonSegment->segmentMatches.empty())
          continue;
        int muonDetector = muonSegment->detector();
        int muonStation = muonSegment->station() - 1;
        assert(muonStation >= 0 && muonStation <= 3);
        if (muonDetector == MuonSubdetId::DT)
          ++numMatchesDT[muonStation];
        else if (muonDetector == MuonSubdetId::CSC)
          ++numMatchesCSC[muonStation];
        else if (muonDetector == MuonSubdetId::RPC)
          ++numMatchesRPC[muonStation];
      }
    }

  }  // namespace tau
}  // namespace reco
