#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionHitChecker.h"
// Framework
//

std::pair<uint8_t, Measurement1DFloat> ConversionHitChecker::nHitsBeforeVtx(const reco::TrackExtra &track,
                                                                            const reco::Vertex &vtx,
                                                                            float sigmaTolerance) const {
  // track hits are always inout

  GlobalPoint vtxPos(vtx.x(), vtx.y(), vtx.z());

  auto const &trajParams = track.trajParams();

  //iterate inside out, when distance to vertex starts increasing, we are at the closest hit
  // the first (and last, btw) hit is always valid... (apparntly not..., conversion is different????)
  TrackingRecHit const *recHit = *track.recHits().begin();
  unsigned int closest = 0;
  for (auto const &hit : track.recHits()) {
    if (hit->isValid()) {
      recHit = hit;
      break;
    }
    ++closest;
  }
  auto globalPosition = recHit->surface()->toGlobal(trajParams[closest].position());
  auto distance2 = (vtxPos - globalPosition).mag2();
  int nhits = 1;
  for (unsigned int h = closest + 1; h < track.recHitsSize(); ++h) {
    //check if next valid hit is farther away from vertex than existing closest
    auto nextHit = track.recHit(h);
    if (!nextHit->isValid())
      continue;
    globalPosition = nextHit->surface()->toGlobal(trajParams[h].position());
    auto nextDistance2 = (vtxPos - globalPosition).mag2();
    if (nextDistance2 > distance2)
      break;

    distance2 = nextDistance2;
    ++nhits;
    closest = h;
  }

  //compute signed decaylength significance for closest hit and check if it is before the vertex
  //if not then we need to subtract it from the count of hits before the vertex, since it has been implicitly included
  recHit = track.recHit(closest).get();
  auto momDir = recHit->surface()->toGlobal(trajParams[closest].direction());
  globalPosition = recHit->surface()->toGlobal(trajParams[closest].position());
  float decayLengthHitToVtx = (vtxPos - globalPosition).dot(momDir);

  AlgebraicVector3 j;
  j[0] = momDir.x();
  j[1] = momDir.y();
  j[2] = momDir.z();
  float vertexError2 = ROOT::Math::Similarity(j, vtx.covariance());
  auto decayLenError = std::sqrt(vertexError2);

  Measurement1DFloat decayLength(decayLengthHitToVtx, decayLenError);

  if (decayLength.significance() <
      sigmaTolerance) {  //decay length is not (significantly) positive, so hit is consistent with the vertex position or late
                         //subtract it from wrong hits count
    --nhits;
  }

  return std::pair<unsigned int, Measurement1DFloat>(nhits, decayLength);
}

uint8_t ConversionHitChecker::nSharedHits(const reco::Track &trk1, const reco::Track &trk2) const {
  uint8_t nShared = 0;

  for (trackingRecHit_iterator iHit1 = trk1.recHitsBegin(); iHit1 != trk1.recHitsEnd(); ++iHit1) {
    const TrackingRecHit *hit1 = (*iHit1);
    if (hit1->isValid()) {
      for (trackingRecHit_iterator iHit2 = trk2.recHitsBegin(); iHit2 != trk2.recHitsEnd(); ++iHit2) {
        const TrackingRecHit *hit2 = (*iHit2);
        if (hit2->isValid() && hit1->sharesInput(hit2, TrackingRecHit::some)) {
          ++nShared;
        }
      }
    }
  }

  return nShared;
}
