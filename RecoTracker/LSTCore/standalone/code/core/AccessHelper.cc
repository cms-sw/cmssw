#include "AccessHelper.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// ===============
// ----* Hit *----
// ===============

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> convertHitsToHitIdxsAndHitTypes(
    lst::Event<Acc3D>* event, std::vector<unsigned int> hits) {
  lst::Hits const* hitsEvt = event->getHits()->data();
  std::vector<unsigned int> hitidxs;
  std::vector<unsigned int> hittypes;
  for (auto& hit : hits) {
    hitidxs.push_back(hitsEvt->idxs[hit]);
    if (hitsEvt->detid[hit] == 1)
      hittypes.push_back(0);
    else
      hittypes.push_back(4);
  }
  return std::make_tuple(hitidxs, hittypes);
}

// ===============
// ----* pLS *----
// ===============

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitsFrompLS(lst::Event<Acc3D>* event, unsigned int pLS) {
  lst::Segments const* segments = event->getSegments()->data();
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::ObjectRanges const* rangesEvt = event->getRanges()->data();
  lst::Modules const* modulesEvt = event->getModules()->data();
  const unsigned int pLS_offset = rangesEvt->segmentModuleIndices[*(modulesEvt->nLowerModules)];
  unsigned int MD_1 = segments->mdIndices[2 * (pLS + pLS_offset)];
  unsigned int MD_2 = segments->mdIndices[2 * (pLS + pLS_offset) + 1];
  unsigned int hit_1 = miniDoublets->anchorHitIndices[MD_1];
  unsigned int hit_2 = miniDoublets->outerHitIndices[MD_1];
  unsigned int hit_3 = miniDoublets->anchorHitIndices[MD_2];
  unsigned int hit_4 = miniDoublets->outerHitIndices[MD_2];
  if (hit_3 == hit_4)
    return {hit_1, hit_2, hit_3};
  else
    return {hit_1, hit_2, hit_3, hit_4};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitIdxsFrompLS(lst::Event<Acc3D>* event, unsigned int pLS) {
  lst::Hits const* hitsEvt = event->getHits()->data();
  std::vector<unsigned int> hits = getPixelHitsFrompLS(event, pLS);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsEvt->idxs[hit]);
  return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitTypesFrompLS(lst::Event<Acc3D>* event, unsigned int pLS) {
  std::vector<unsigned int> hits = getPixelHitsFrompLS(event, pLS);
  std::vector<unsigned int> hittypes(hits.size(), 0);
  return hittypes;
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompLS(lst::Event<Acc3D>* event,
                                                                                              unsigned pLS) {
  return convertHitsToHitIdxsAndHitTypes(event, getPixelHitsFrompLS(event, pLS));
}

// ==============
// ----* MD *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromMD(lst::Event<Acc3D>* event, unsigned int MD) {
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  unsigned int hit_1 = miniDoublets->anchorHitIndices[MD];
  unsigned int hit_2 = miniDoublets->outerHitIndices[MD];
  return {hit_1, hit_2};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromMD(lst::Event<Acc3D>* event,
                                                                                             unsigned MD) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromMD(event, MD));
}

// ==============
// ----* LS *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromLS(lst::Event<Acc3D>* event, unsigned int LS) {
  lst::Segments const* segments = event->getSegments()->data();
  unsigned int MD_1 = segments->mdIndices[2 * LS];
  unsigned int MD_2 = segments->mdIndices[2 * LS + 1];
  return {MD_1, MD_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromLS(lst::Event<Acc3D>* event, unsigned int LS) {
  std::vector<unsigned int> MDs = getMDsFromLS(event, LS);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1]};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromLS(lst::Event<Acc3D>* event,
                                                                                             unsigned LS) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromLS(event, LS));
}

// ==============
// ----* T3 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromT3(lst::Event<Acc3D>* event, unsigned int T3) {
  lst::Triplets const* triplets = event->getTriplets()->data();
  unsigned int LS_1 = triplets->segmentIndices[2 * T3];
  unsigned int LS_2 = triplets->segmentIndices[2 * T3 + 1];
  return {LS_1, LS_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromT3(lst::Event<Acc3D>* event, unsigned int T3) {
  std::vector<unsigned int> LSs = getLSsFromT3(event, T3);
  std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
  std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
  return {MDs_0[0], MDs_0[1], MDs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromT3(lst::Event<Acc3D>* event, unsigned int T3) {
  std::vector<unsigned int> MDs = getMDsFromT3(event, T3);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1]};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromT3(lst::Event<Acc3D>* event,
                                                                                             unsigned T3) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromT3(event, T3));
}

// ==============
// ----* T5 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getT3sFromT5(lst::Event<Acc3D>* event, unsigned int T5) {
  lst::Quintuplets const* quintuplets = event->getQuintuplets()->data();
  unsigned int T3_1 = quintuplets->tripletIndices[2 * T5];
  unsigned int T3_2 = quintuplets->tripletIndices[2 * T5 + 1];
  return {T3_1, T3_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromT5(lst::Event<Acc3D>* event, unsigned int T5) {
  std::vector<unsigned int> T3s = getT3sFromT5(event, T5);
  std::vector<unsigned int> LSs_0 = getLSsFromT3(event, T3s[0]);
  std::vector<unsigned int> LSs_1 = getLSsFromT3(event, T3s[1]);
  return {LSs_0[0], LSs_0[1], LSs_1[0], LSs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromT5(lst::Event<Acc3D>* event, unsigned int T5) {
  std::vector<unsigned int> LSs = getLSsFromT5(event, T5);
  std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
  std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
  std::vector<unsigned int> MDs_2 = getMDsFromLS(event, LSs[2]);
  std::vector<unsigned int> MDs_3 = getMDsFromLS(event, LSs[3]);
  return {MDs_0[0], MDs_0[1], MDs_1[1], MDs_2[1], MDs_3[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromT5(lst::Event<Acc3D>* event, unsigned int T5) {
  std::vector<unsigned int> MDs = getMDsFromT5(event, T5);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
  std::vector<unsigned int> hits_3 = getHitsFromMD(event, MDs[3]);
  std::vector<unsigned int> hits_4 = getHitsFromMD(event, MDs[4]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1], hits_3[0], hits_3[1], hits_4[0], hits_4[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFromT5(lst::Event<Acc3D>* event, unsigned int T5) {
  lst::Hits const* hitsEvt = event->getHits()->data();
  std::vector<unsigned int> hits = getHitsFromT5(event, T5);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsEvt->idxs[hit]);
  return hitidxs;
}
//____________________________________________________________________________________________
std::vector<unsigned int> getModuleIdxsFromT5(lst::Event<Acc3D>* event, unsigned int T5) {
  std::vector<unsigned int> hits = getHitsFromT5(event, T5);
  std::vector<unsigned int> module_idxs;
  lst::Hits const* hitsEvt = event->getHits()->data();
  for (auto& hitIdx : hits) {
    module_idxs.push_back(hitsEvt->moduleIndices[hitIdx]);
  }
  return module_idxs;
}
//____________________________________________________________________________________________
std::vector<unsigned int> getHitTypesFromT5(lst::Event<Acc3D>* event, unsigned int T5) {
  return {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  ;
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromT5(lst::Event<Acc3D>* event,
                                                                                             unsigned T5) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromT5(event, T5));
}

// ===============
// ----* pT3 *----
// ===============

//____________________________________________________________________________________________
unsigned int getPixelLSFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  lst::PixelTriplets const* pixelTriplets = event->getPixelTriplets()->data();
  lst::ObjectRanges const* rangesEvt = event->getRanges()->data();
  lst::Modules const* modulesEvt = event->getModules()->data();
  const unsigned int pLS_offset = rangesEvt->segmentModuleIndices[*(modulesEvt->nLowerModules)];
  return pixelTriplets->pixelSegmentIndices[pT3] - pLS_offset;
}

//____________________________________________________________________________________________
unsigned int getT3FrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  lst::PixelTriplets const* pixelTriplets = event->getPixelTriplets()->data();
  return pixelTriplets->tripletIndices[pT3];
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getLSsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getMDsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getOuterTrackerHitsFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getHitsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitsFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  unsigned int pLS = getPixelLSFrompT3(event, pT3);
  return getPixelHitsFrompLS(event, pLS);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  unsigned int pLS = getPixelLSFrompT3(event, pT3);
  unsigned int T3 = getT3FrompT3(event, pT3);
  std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
  std::vector<unsigned int> outerTrackerHits = getHitsFromT3(event, T3);
  pixelHits.insert(pixelHits.end(), outerTrackerHits.begin(), outerTrackerHits.end());
  return pixelHits;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  lst::Hits const* hitsEvt = event->getHits()->data();
  std::vector<unsigned int> hits = getHitsFrompT3(event, pT3);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsEvt->idxs[hit]);
  return hitidxs;
}
//____________________________________________________________________________________________
std::vector<unsigned int> getModuleIdxsFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  std::vector<unsigned int> hits = getOuterTrackerHitsFrompT3(event, pT3);
  std::vector<unsigned int> module_idxs;
  lst::Hits const* hitsEvt = event->getHits()->data();
  for (auto& hitIdx : hits) {
    module_idxs.push_back(hitsEvt->moduleIndices[hitIdx]);
  }
  return module_idxs;
}
//____________________________________________________________________________________________
std::vector<unsigned int> getHitTypesFrompT3(lst::Event<Acc3D>* event, unsigned int pT3) {
  unsigned int pLS = getPixelLSFrompT3(event, pT3);
  std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
  // pixel Hits list will be either 3 or 4 and depending on it return accordingly
  if (pixelHits.size() == 3)
    return {0, 0, 0, 4, 4, 4, 4, 4, 4};
  else
    return {0, 0, 0, 0, 4, 4, 4, 4, 4, 4};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompT3(lst::Event<Acc3D>* event,
                                                                                              unsigned pT3) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFrompT3(event, pT3));
}

// ===============
// ----* pT5 *----
// ===============

//____________________________________________________________________________________________
unsigned int getPixelLSFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  lst::PixelQuintuplets const* pixelQuintuplets = event->getPixelQuintuplets()->data();
  lst::ObjectRanges const* rangesEvt = event->getRanges()->data();
  lst::Modules const* modulesEvt = event->getModules()->data();
  const unsigned int pLS_offset = rangesEvt->segmentModuleIndices[*(modulesEvt->nLowerModules)];
  return pixelQuintuplets->pixelIndices[pT5] - pLS_offset;
}

//____________________________________________________________________________________________
unsigned int getT5FrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  lst::PixelQuintuplets const* pixelQuintuplets = event->getPixelQuintuplets()->data();
  return pixelQuintuplets->T5Indices[pT5];
}

//____________________________________________________________________________________________
std::vector<unsigned int> getT3sFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getT3sFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getLSsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getMDsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getOuterTrackerHitsFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getHitsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitsFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  unsigned int pLS = getPixelLSFrompT5(event, pT5);
  return getPixelHitsFrompLS(event, pLS);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  unsigned int pLS = getPixelLSFrompT5(event, pT5);
  unsigned int T5 = getT5FrompT5(event, pT5);
  std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
  std::vector<unsigned int> outerTrackerHits = getHitsFromT5(event, T5);
  pixelHits.insert(pixelHits.end(), outerTrackerHits.begin(), outerTrackerHits.end());
  return pixelHits;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  lst::Hits const* hitsEvt = event->getHits()->data();
  std::vector<unsigned int> hits = getHitsFrompT5(event, pT5);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsEvt->idxs[hit]);
  return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getModuleIdxsFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  std::vector<unsigned int> hits = getOuterTrackerHitsFrompT5(event, pT5);
  std::vector<unsigned int> module_idxs;
  lst::Hits const* hitsEvt = event->getHits()->data();
  for (auto& hitIdx : hits) {
    module_idxs.push_back(hitsEvt->moduleIndices[hitIdx]);
  }
  return module_idxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitTypesFrompT5(lst::Event<Acc3D>* event, unsigned int pT5) {
  unsigned int pLS = getPixelLSFrompT5(event, pT5);
  std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
  // pixel Hits list will be either 3 or 4 and depending on it return accordingly
  if (pixelHits.size() == 3)
    return {0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  else
    return {0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompT5(lst::Event<Acc3D>* event,
                                                                                              unsigned pT5) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFrompT5(event, pT5));
}

// ==============
// ----* TC *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromTC(lst::Event<Acc3D>* event, unsigned int TC) {
  // Get the type of the track candidate
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  short type = trackCandidates->trackCandidateType[TC];
  unsigned int objidx = trackCandidates->directObjectIndices[TC];
  switch (type) {
    case kpT5:
      return getLSsFrompT5(event, objidx);
      break;
    case kpT3:
      return getLSsFrompT3(event, objidx);
      break;
    case kT5:
      return getLSsFromT5(event, objidx);
      break;
    case kpLS:
      return std::vector<unsigned int>();
      break;
  }
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromTC(lst::Event<Acc3D>* event,
                                                                                             unsigned TC) {
  // Get the type of the track candidate
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  short type = trackCandidates->trackCandidateType[TC];
  unsigned int objidx = trackCandidates->directObjectIndices[TC];
  switch (type) {
    case kpT5:
      return getHitIdxsAndHitTypesFrompT5(event, objidx);
      break;
    case kpT3:
      return getHitIdxsAndHitTypesFrompT3(event, objidx);
      break;
    case kT5:
      return getHitIdxsAndHitTypesFromT5(event, objidx);
      break;
    case kpLS:
      return getHitIdxsAndHitTypesFrompLS(event, objidx);
      break;
  }
}
