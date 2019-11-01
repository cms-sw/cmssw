#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronTools.h"

namespace egamma {

  using namespace reco;

  //=======================================================================================
  // Code from Puneeth Kalavase
  //=======================================================================================

  std::pair<TrackRef, float> getClosestCtfToGsf(GsfTrackRef const& gsfTrackRef,
                                                edm::Handle<reco::TrackCollection> const& ctfTracksH) {
    float maxFracShared = 0;
    TrackRef ctfTrackRef = TrackRef();
    const TrackCollection* ctfTrackCollection = ctfTracksH.product();

    // get the Hit Pattern for the gsfTrack
    const HitPattern& gsfHitPattern = gsfTrackRef->hitPattern();

    unsigned int counter = 0;
    for (auto ctfTkIter = ctfTrackCollection->begin(); ctfTkIter != ctfTrackCollection->end(); ctfTkIter++, counter++) {
      double dEta = gsfTrackRef->eta() - ctfTkIter->eta();
      double dPhi = gsfTrackRef->phi() - ctfTkIter->phi();
      double pi = acos(-1.);
      if (std::abs(dPhi) > pi)
        dPhi = 2 * pi - std::abs(dPhi);

      // dont want to look at every single track in the event!
      if (sqrt(dEta * dEta + dPhi * dPhi) > 0.3)
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

    return make_pair(ctfTrackRef, maxFracShared);
  }

}  // namespace egamma
