#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"
namespace PFTrackAlgoTools {

  double dPtCut(const reco::TrackBase::TrackAlgorithm& algo,const std::vector<double>& cuts,bool hltIterativeTracking = true){
    switch (algo) {
    case reco::TrackBase::ctf:
    case reco::TrackBase::duplicateMerge:
    case reco::TrackBase::initialStep:
    case reco::TrackBase::highPtTripletStep:
    case reco::TrackBase::lowPtQuadStep:
    case reco::TrackBase::lowPtTripletStep:
    case reco::TrackBase::pixelPairStep:
    case reco::TrackBase::jetCoreRegionalStep:
      return cuts[0];
    case reco::TrackBase::detachedQuadStep:
    case reco::TrackBase::detachedTripletStep:
      return cuts[1];
    case reco::TrackBase::mixedTripletStep:
      return cuts[2];
    case reco::TrackBase::pixelLessStep:
      return cuts[3];
    case reco::TrackBase::tobTecStep:
      return cuts[4];
    case reco::TrackBase::muonSeededStepInOut:
    case reco::TrackBase::muonSeededStepOutIn:
      return cuts[5];
    case reco::TrackBase::hltIter0:
    case reco::TrackBase::hltIter1:
    case reco::TrackBase::hltIter2:
      return cuts[0];
    case reco::TrackBase::hltIter3:
      return  hltIterativeTracking ? cuts[1] : cuts[0];
    case reco::TrackBase::hltIter4:
      return  hltIterativeTracking ? cuts[2] : cuts[0];
    case reco::TrackBase::hltIterX:
      return  cuts[0];
    default:
      return hltIterativeTracking ? cuts[6]:cuts[0];

    }
  }



  unsigned int  nHitCut(const reco::TrackBase::TrackAlgorithm& algo,const std::vector<unsigned int>& cuts,bool hltIterativeTracking = true){
    switch (algo) {
    case reco::TrackBase::ctf:
    case reco::TrackBase::duplicateMerge:
    case reco::TrackBase::initialStep:
    case reco::TrackBase::highPtTripletStep:
    case reco::TrackBase::lowPtQuadStep:
    case reco::TrackBase::lowPtTripletStep:
    case reco::TrackBase::pixelPairStep:
    case reco::TrackBase::jetCoreRegionalStep:
      return cuts[0];
    case reco::TrackBase::detachedQuadStep:
    case reco::TrackBase::detachedTripletStep:
      return cuts[1];
    case reco::TrackBase::mixedTripletStep:
      return cuts[2];
    case reco::TrackBase::pixelLessStep:
      return cuts[3];
    case reco::TrackBase::tobTecStep:
      return cuts[4];
    case reco::TrackBase::muonSeededStepInOut:
    case reco::TrackBase::muonSeededStepOutIn:
      return cuts[5];
    case reco::TrackBase::hltIter0:
    case reco::TrackBase::hltIter1:
    case reco::TrackBase::hltIter2:
      return cuts[0];
    case reco::TrackBase::hltIter3:
      return  hltIterativeTracking ? cuts[1] : cuts[0];
    case reco::TrackBase::hltIter4:
      return  hltIterativeTracking ? cuts[2] : cuts[0];
    case reco::TrackBase::hltIterX:
      return  cuts[0];
    default:
      return hltIterativeTracking ? cuts[6]:cuts[0];

    }
  }



  double  errorScale(const reco::TrackBase::TrackAlgorithm& algo,const std::vector<double>& errorScale){
    switch (algo) {
    case reco::TrackBase::ctf:
    case reco::TrackBase::duplicateMerge:
    case reco::TrackBase::initialStep:
    case reco::TrackBase::highPtTripletStep:
    case reco::TrackBase::lowPtQuadStep:
    case reco::TrackBase::lowPtTripletStep:
    case reco::TrackBase::pixelPairStep:
    case reco::TrackBase::jetCoreRegionalStep:
    case reco::TrackBase::muonSeededStepInOut:
    case reco::TrackBase::muonSeededStepOutIn:
    case reco::TrackBase::detachedQuadStep:
    case reco::TrackBase::detachedTripletStep:
    case reco::TrackBase::mixedTripletStep:
    case reco::TrackBase::hltIter0:
    case reco::TrackBase::hltIter1:
    case reco::TrackBase::hltIter2:
    case reco::TrackBase::hltIter3:
    case reco::TrackBase::hltIter4:
    case reco::TrackBase::hltIterX:
      return 1.0;
    case reco::TrackBase::pixelLessStep:
      return errorScale[0];
    case reco::TrackBase::tobTecStep:
      return errorScale[1];
    default:
      return 1E9;
    }
  }


bool isGoodForEGM(const reco::TrackBase::TrackAlgorithm& algo){


  switch (algo) {
  case reco::TrackBase::ctf:
  case reco::TrackBase::duplicateMerge:
  case reco::TrackBase::initialStep:
  case reco::TrackBase::highPtTripletStep:
  case reco::TrackBase::lowPtQuadStep:
  case reco::TrackBase::lowPtTripletStep:
  case reco::TrackBase::pixelPairStep:
  case reco::TrackBase::jetCoreRegionalStep:
  case reco::TrackBase::detachedQuadStep:
  case reco::TrackBase::detachedTripletStep:
  case reco::TrackBase::mixedTripletStep:
  case reco::TrackBase::muonSeededStepInOut:
  case reco::TrackBase::muonSeededStepOutIn:
  case reco::TrackBase::hltIter0:
  case reco::TrackBase::hltIter1:
  case reco::TrackBase::hltIter2:
  case reco::TrackBase::hltIter3:
  case reco::TrackBase::hltIter4:
  case reco::TrackBase::hltIterX:
    return true;
  default:
    return false;
  }

}

bool isGoodForEGMPrimary(const reco::TrackBase::TrackAlgorithm& algo){
  switch (algo) {
  case reco::TrackBase::ctf:
  case reco::TrackBase::duplicateMerge:
  case reco::TrackBase::cosmics:
  case reco::TrackBase::initialStep:
  case reco::TrackBase::highPtTripletStep:
  case reco::TrackBase::lowPtQuadStep:
  case reco::TrackBase::lowPtTripletStep:
  case reco::TrackBase::pixelPairStep:
  case reco::TrackBase::detachedQuadStep:
  case reco::TrackBase::detachedTripletStep:
  case reco::TrackBase::mixedTripletStep:
    return true;
  default:
    return false;
  }

}

bool isFifthStep(const reco::TrackBase::TrackAlgorithm& algo){
  switch (algo) {
  case reco::TrackBase::ctf:
  case reco::TrackBase::duplicateMerge:
  case reco::TrackBase::initialStep:
  case reco::TrackBase::highPtTripletStep:
  case reco::TrackBase::lowPtQuadStep:
  case reco::TrackBase::lowPtTripletStep:
  case reco::TrackBase::pixelPairStep:
  case reco::TrackBase::jetCoreRegionalStep:
  case reco::TrackBase::detachedQuadStep:
  case reco::TrackBase::detachedTripletStep:
  case reco::TrackBase::mixedTripletStep:
  case reco::TrackBase::pixelLessStep:
  case reco::TrackBase::muonSeededStepInOut:
  case reco::TrackBase::muonSeededStepOutIn:
  case reco::TrackBase::hltIter0:
  case reco::TrackBase::hltIter1:
  case reco::TrackBase::hltIter2:
  case reco::TrackBase::hltIter3:
  case reco::TrackBase::hltIter4:
  case reco::TrackBase::hltIterX:
    return false;
  case reco::TrackBase::tobTecStep:
    return true;
  default:
    return true;
  }

}


bool highQuality(const reco::TrackBase::TrackAlgorithm& algo){
  switch (algo) {
  case reco::TrackBase::initialStep:
  case reco::TrackBase::highPtTripletStep:
  case reco::TrackBase::lowPtQuadStep:
  case reco::TrackBase::lowPtTripletStep:
  case reco::TrackBase::pixelPairStep:
  case reco::TrackBase::detachedQuadStep:
  case reco::TrackBase::detachedTripletStep:
  case reco::TrackBase::duplicateMerge:
  case reco::TrackBase::jetCoreRegionalStep:
    return true;
  default:
    return false;
  }

}


bool nonIterative(const reco::TrackBase::TrackAlgorithm& algo){
  switch (algo) {
  case reco::TrackBase::undefAlgorithm:
  case reco::TrackBase::ctf:
  case reco::TrackBase::cosmics:
    return true;
  default:
    return false;

  }

}


bool step45(const reco::TrackBase::TrackAlgorithm& algo){
  switch (algo) {
  case reco::TrackBase::mixedTripletStep:
  case reco::TrackBase::pixelLessStep:
  case reco::TrackBase::tobTecStep:
    return true;
  default:
    return false;
  }

}


bool step5(const reco::TrackBase::TrackAlgorithm& algo){
  return (algo==reco::TrackBase::tobTecStep||algo==reco::TrackBase::pixelLessStep);
}

}
