#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"
namespace PFTrackAlgoTools {
unsigned int getAlgoCategory(const reco::TrackBase::TrackAlgorithm& algo,bool hltIterativeTracking) {
  switch (algo) {
  case reco::TrackBase::ctf:
  case reco::TrackBase::duplicateMerge:
  case reco::TrackBase::initialStep:
  case reco::TrackBase::lowPtTripletStep:
  case reco::TrackBase::pixelPairStep:
  case reco::TrackBase::jetCoreRegionalStep:
    return 0;
  case reco::TrackBase::detachedTripletStep:
    return 1;
  case reco::TrackBase::mixedTripletStep:
    return 2;
  case reco::TrackBase::pixelLessStep:
    return 3;
  case reco::TrackBase::tobTecStep:
    return 4;
  case reco::TrackBase::muonSeededStepInOut:
  case reco::TrackBase::muonSeededStepOutIn:
    return 5;
  case reco::TrackBase::hltIter0:
  case reco::TrackBase::hltIter1:
  case reco::TrackBase::hltIter2:
    return 0;
  case reco::TrackBase::hltIter3:
    return  hltIterativeTracking ? 1 : 0;
  case reco::TrackBase::hltIter4:
    return  hltIterativeTracking ? 2 : 0;
  case reco::TrackBase::hltIterX:
    return  0;
  default:
    return hltIterativeTracking ? 6:0;
  }

}
}
