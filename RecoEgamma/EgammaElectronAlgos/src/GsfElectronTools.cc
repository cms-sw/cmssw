#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronTools.h"

namespace egamma {

  std::vector<TrackVariables> getTrackVariables(reco::TrackCollection const& tracks) {
    std::vector<TrackVariables> out;
    out.reserve(tracks.size());
    for (auto const& tk : tracks) {
      out.emplace_back(tk);
    }
    return out;
  }

  using namespace reco;

  //=======================================================================================
  // Code from Puneeth Kalavase
  //=======================================================================================

  std::pair<TrackRef, float> getClosestCtfToGsf(GsfTrackRef const& gsfTrackRef,
                                                edm::Handle<reco::TrackCollection> const& ctfTracksH,
                                                std::vector<TrackVariables> const& ctfTrackVariables) {
    float maxFracShared = 0;
    TrackRef ctfTrackRef = TrackRef();
    const TrackCollection* ctfTrackCollection = ctfTracksH.product();

    double gsfEta = gsfTrackRef->eta();
    double gsfPhi = gsfTrackRef->phi();
    const HitPattern& gsfHitPattern = gsfTrackRef->hitPattern();

    constexpr double dR2 = 0.3 * 0.3;

    unsigned int counter = 0;
    for (auto ctfTkIter = ctfTrackCollection->begin(); ctfTkIter != ctfTrackCollection->end(); ctfTkIter++, counter++) {
      auto const& trackVariables = ctfTrackVariables[counter];
      double dEta = gsfEta - trackVariables.eta;
      double dPhi = gsfPhi - trackVariables.phi;
      if (std::abs(dPhi) > M_PI)
        dPhi = 2 * M_PI - std::abs(dPhi);

      // dont want to look at every single track in the event!
      if (dEta * dEta + dPhi * dPhi > dR2)
        continue;

      unsigned int shared = 0;
      int gsfHitCounter = 0;
      int numGsfInnerHits = 0;
      int numCtfInnerHits = 0;
      // get the CTF Track Hit Pattern
      const HitPattern& ctfHitPattern = ctfTkIter->hitPattern();

      for (auto elHitsIt = gsfTrackRef->recHitsBegin(); elHitsIt != gsfTrackRef->recHitsEnd();
           elHitsIt++, gsfHitCounter++) {
        if (!((**elHitsIt).isValid()))  //count only valid Hits
        {
          continue;
        }

        // look only in the pixels/TIB/TID
        uint32_t gsfHit = gsfHitPattern.getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter);
        if (!(HitPattern::pixelHitFilter(gsfHit) || HitPattern::stripTIBHitFilter(gsfHit) ||
              HitPattern::stripTIDHitFilter(gsfHit))) {
          continue;
        }

        numGsfInnerHits++;

        int ctfHitsCounter = 0;
        numCtfInnerHits = 0;
        for (auto ctfHitsIt = ctfTkIter->recHitsBegin(); ctfHitsIt != ctfTkIter->recHitsEnd();
             ctfHitsIt++, ctfHitsCounter++) {
          if (!((**ctfHitsIt).isValid()))  //count only valid Hits!
          {
            continue;
          }

          uint32_t ctfHit = ctfHitPattern.getHitPattern(HitPattern::TRACK_HITS, ctfHitsCounter);
          if (!(HitPattern::pixelHitFilter(ctfHit) || HitPattern::stripTIBHitFilter(ctfHit) ||
                HitPattern::stripTIDHitFilter(ctfHit))) {
            continue;
          }

          numCtfInnerHits++;

          if ((**elHitsIt).sharesInput(&(**ctfHitsIt), TrackingRecHit::all)) {
            shared++;
            break;
          }

        }  //ctfHits iterator

      }  //gsfHits iterator

      if ((numGsfInnerHits == 0) || (numCtfInnerHits == 0)) {
        continue;
      }

      if (static_cast<float>(shared) / std::min(numGsfInnerHits, numCtfInnerHits) > maxFracShared) {
        maxFracShared = static_cast<float>(shared) / std::min(numGsfInnerHits, numCtfInnerHits);
        ctfTrackRef = TrackRef(ctfTracksH, counter);
      }

    }  //ctfTrack iterator

    return {ctfTrackRef, maxFracShared};
  }

}  // namespace egamma
