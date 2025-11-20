#include "AccessHelper.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

// ===============
// ----* Hit *----
// ===============

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> convertHitsToHitIdxsAndHitTypes(
    LSTEvent* event, std::vector<unsigned int> hits) {
  auto hitsBase = event->getInput<HitsBaseSoA>();
  std::vector<unsigned int> hitidxs;
  std::vector<HitType> hittypes;
  for (auto& hit : hits) {
    hitidxs.push_back(hitsBase.idxs()[hit]);
    if (hitsBase.detid()[hit] == kPixelModuleId)
      hittypes.push_back(HitType::Pixel);
    else
      hittypes.push_back(HitType::Phase2OT);
  }
  return std::make_tuple(hitidxs, hittypes);
}

// ===============
// ----* pLS *----
// ===============

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFrompLS(LSTEvent* event, unsigned int pLS) {
  SegmentsConst segments = event->getSegments<SegmentsSoA>();
  MiniDoubletsConst miniDoublets = event->getMiniDoublets<MiniDoubletsSoA>();
  auto ranges = event->getRanges();
  auto modulesEvt = event->getModules<ModulesSoA>();
  const unsigned int pLS_offset = ranges.segmentModuleIndices()[modulesEvt.nLowerModules()];
  unsigned int MD_1 = segments.mdIndices()[pLS + pLS_offset][0];
  unsigned int MD_2 = segments.mdIndices()[pLS + pLS_offset][1];
  unsigned int hit_1 = miniDoublets.anchorHitIndices()[MD_1];
  unsigned int hit_2 = miniDoublets.outerHitIndices()[MD_1];
  unsigned int hit_3 = miniDoublets.anchorHitIndices()[MD_2];
  unsigned int hit_4 = miniDoublets.outerHitIndices()[MD_2];
  if (hit_3 == hit_4)
    return {hit_1, hit_2, hit_3};
  else
    return {hit_1, hit_2, hit_3, hit_4};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFrompLS(LSTEvent* event, unsigned int pLS) {
  auto hitsBase = event->getInput<HitsBaseSoA>();
  std::vector<unsigned int> hits = getHitsFrompLS(event, pLS);
  std::vector<unsigned int> hitidxs;
  hitidxs.reserve(hits.size());
  for (auto& hit : hits)
    hitidxs.push_back(hitsBase.idxs()[hit]);
  return hitidxs;
}

//____________________________________________________________________________________________
std::vector<HitType> getHitTypesFrompLS(LSTEvent* event, unsigned int pLS) {
  std::vector<unsigned int> hits = getHitsFrompLS(event, pLS);
  std::vector<HitType> hittypes;
  hittypes.reserve(hits.size());
  auto hitsBase = event->getInput<HitsBaseSoA>();
  for (auto& hit : hits)
    hittypes.push_back(hitsBase.detid()[hit] == kPixelModuleId ? HitType::Pixel : HitType::Phase2OT);
  return hittypes;
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFrompLS(LSTEvent* event,
                                                                                         unsigned pLS) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFrompLS(event, pLS));
}

// ==============
// ----* MD *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromMD(LSTEvent* event, unsigned int MD) {
  MiniDoubletsConst miniDoublets = event->getMiniDoublets<MiniDoubletsSoA>();
  unsigned int hit_1 = miniDoublets.anchorHitIndices()[MD];
  unsigned int hit_2 = miniDoublets.outerHitIndices()[MD];
  return {hit_1, hit_2};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFromMD(LSTEvent* event, unsigned MD) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromMD(event, MD));
}

// ==============
// ----* LS *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromLS(LSTEvent* event, unsigned int LS) {
  SegmentsConst segments = event->getSegments<SegmentsSoA>();
  unsigned int MD_1 = segments.mdIndices()[LS][0];
  unsigned int MD_2 = segments.mdIndices()[LS][1];
  return {MD_1, MD_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromLS(LSTEvent* event, unsigned int LS) {
  std::vector<unsigned int> MDs = getMDsFromLS(event, LS);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1]};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFromLS(LSTEvent* event, unsigned LS) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromLS(event, LS));
}

// ==============
// ----* T3 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromT3(LSTEvent* event, unsigned int t3) {
  auto const triplets = event->getTriplets<TripletsSoA>();
  unsigned int ls_1 = triplets.segmentIndices()[t3][0];
  unsigned int ls_2 = triplets.segmentIndices()[t3][1];
  return {ls_1, ls_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromT3(LSTEvent* event, unsigned int T3) {
  std::vector<unsigned int> LSs = getLSsFromT3(event, T3);
  std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
  std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
  return {MDs_0[0], MDs_0[1], MDs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromT3(LSTEvent* event, unsigned int T3) {
  std::vector<unsigned int> MDs = getMDsFromT3(event, T3);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1]};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFromT3(LSTEvent* event, unsigned T3) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromT3(event, T3));
}

// ==============
// ----* T4 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getT3sFromT4(LSTEvent* event, unsigned int t4) {
  auto const quadruplets = event->getQuadruplets<QuadrupletsSoA>();
  unsigned int t3_1 = quadruplets.tripletIndices()[t4][0];
  unsigned int t3_2 = quadruplets.tripletIndices()[t4][1];
  return {t3_1, t3_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromT4(LSTEvent* event, unsigned int T4) {
  std::vector<unsigned int> T3s = getT3sFromT4(event, T4);
  std::vector<unsigned int> LSs_0 = getLSsFromT3(event, T3s[0]);
  std::vector<unsigned int> LSs_1 = getLSsFromT3(event, T3s[1]);
  return {LSs_0[0], LSs_0[1], LSs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromT4(LSTEvent* event, unsigned int T4) {
  std::vector<unsigned int> LSs = getLSsFromT4(event, T4);
  std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
  std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
  std::vector<unsigned int> MDs_2 = getMDsFromLS(event, LSs[2]);
  return {MDs_0[0], MDs_0[1], MDs_2[0], MDs_2[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromT4(LSTEvent* event, unsigned int T4) {
  std::vector<unsigned int> MDs = getMDsFromT4(event, T4);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
  std::vector<unsigned int> hits_3 = getHitsFromMD(event, MDs[3]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1], hits_3[0], hits_3[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFromT4(LSTEvent* event, unsigned int T4) {
  auto hitsBase = event->getInput<HitsBaseSoA>();
  std::vector<unsigned int> hits = getHitsFromT4(event, T4);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsBase.idxs()[hit]);
  return hitidxs;
}
//____________________________________________________________________________________________
std::vector<unsigned int> getModuleIdxsFromT4(LSTEvent* event, unsigned int T4) {
  std::vector<unsigned int> hits = getHitsFromT4(event, T4);
  std::vector<unsigned int> module_idxs;
  auto hitsEvt = event->getHits<HitsExtendedSoA>();
  for (auto& hitIdx : hits) {
    module_idxs.push_back(hitsEvt.moduleIndices()[hitIdx]);
  }
  return module_idxs;
}
//____________________________________________________________________________________________
std::vector<HitType> getHitTypesFromT4(LSTEvent* event, unsigned int T4) {
  return {Params_T4::kHits, HitType::Phase2OT};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFromT4(LSTEvent* event, unsigned T4) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromT4(event, T4));
}

// ==============
// ----* T5 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getT3sFromT5(LSTEvent* event, unsigned int t5) {
  auto const quintuplets = event->getQuintuplets<QuintupletsSoA>();
  unsigned int t3_1 = quintuplets.tripletIndices()[t5][0];
  unsigned int t3_2 = quintuplets.tripletIndices()[t5][1];
  return {t3_1, t3_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromT5(LSTEvent* event, unsigned int T5) {
  std::vector<unsigned int> T3s = getT3sFromT5(event, T5);
  std::vector<unsigned int> LSs_0 = getLSsFromT3(event, T3s[0]);
  std::vector<unsigned int> LSs_1 = getLSsFromT3(event, T3s[1]);
  return {LSs_0[0], LSs_0[1], LSs_1[0], LSs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromT5(LSTEvent* event, unsigned int T5) {
  std::vector<unsigned int> LSs = getLSsFromT5(event, T5);
  std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
  std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
  std::vector<unsigned int> MDs_2 = getMDsFromLS(event, LSs[2]);
  std::vector<unsigned int> MDs_3 = getMDsFromLS(event, LSs[3]);
  return {MDs_0[0], MDs_0[1], MDs_1[1], MDs_2[1], MDs_3[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromT5(LSTEvent* event, unsigned int T5) {
  std::vector<unsigned int> MDs = getMDsFromT5(event, T5);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
  std::vector<unsigned int> hits_3 = getHitsFromMD(event, MDs[3]);
  std::vector<unsigned int> hits_4 = getHitsFromMD(event, MDs[4]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1], hits_3[0], hits_3[1], hits_4[0], hits_4[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFromT5(LSTEvent* event, unsigned int T5) {
  auto hitsBase = event->getInput<HitsBaseSoA>();
  std::vector<unsigned int> hits = getHitsFromT5(event, T5);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsBase.idxs()[hit]);
  return hitidxs;
}
//____________________________________________________________________________________________
std::vector<unsigned int> getModuleIdxsFromT5(LSTEvent* event, unsigned int T5) {
  std::vector<unsigned int> hits = getHitsFromT5(event, T5);
  std::vector<unsigned int> module_idxs;
  auto hitsEvt = event->getHits<HitsExtendedSoA>();
  for (auto& hitIdx : hits) {
    module_idxs.push_back(hitsEvt.moduleIndices()[hitIdx]);
  }
  return module_idxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getModuleIdxsFromT3(LSTEvent* event, unsigned int T3) {
  std::vector<unsigned int> hits = getHitsFromT3(event, T3);
  std::vector<unsigned int> module_idxs;
  auto hitsEvt = event->getHits<HitsExtendedSoA>();
  for (auto& hitIdx : hits) {
    module_idxs.push_back(hitsEvt.moduleIndices()[hitIdx]);
  }
  return module_idxs;
}

//____________________________________________________________________________________________
std::vector<HitType> getHitTypesFromT5(LSTEvent* event, unsigned int T5) {
  return {Params_T5::kHits, HitType::Phase2OT};
}

//____________________________________________________________________________________________
std::vector<HitType> getHitTypesFromT3(LSTEvent* event, unsigned int T5) {
  return {Params_T3::kHits, HitType::Phase2OT};
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFromT5(LSTEvent* event, unsigned T5) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFromT5(event, T5));
}

// ===============
// ----* pT3 *----
// ===============

//____________________________________________________________________________________________
unsigned int getpLSFrompT3(LSTEvent* event, unsigned int pT3) {
  auto const pixelTriplets = event->getPixelTriplets();
  auto ranges = event->getRanges();
  auto modulesEvt = event->getModules<ModulesSoA>();
  const unsigned int pLS_offset = ranges.segmentModuleIndices()[modulesEvt.nLowerModules()];
  return pixelTriplets.pixelSegmentIndices()[pT3] - pLS_offset;
}

//____________________________________________________________________________________________
unsigned int getT3FrompT3(LSTEvent* event, unsigned int pT3) {
  auto const pixelTriplets = event->getPixelTriplets();
  return pixelTriplets.tripletIndices()[pT3];
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFrompT3(LSTEvent* event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getLSsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFrompT3(LSTEvent* event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getMDsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getT3HitsFrompT3(LSTEvent* event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getHitsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getpLSHitsFrompT3(LSTEvent* event, unsigned int pT3) {
  unsigned int pLS = getpLSFrompT3(event, pT3);
  return getHitsFrompLS(event, pLS);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFrompT3(LSTEvent* event, unsigned int pT3) {
  auto const& allHits = event->getPixelTriplets().hitIndices()[pT3];
  std::vector<unsigned int> hits;
  hits.reserve(allHits.size());
  for (auto const hit : allHits)
    if (hits.empty() || hits.back() != hit)  // should eventually check all and type
      hits.emplace_back(hit);
  return hits;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFrompT3(LSTEvent* event, unsigned int pT3) {
  auto hitsBase = event->getInput<HitsBaseSoA>();
  std::vector<unsigned int> hits = getHitsFrompT3(event, pT3);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsBase.idxs()[hit]);
  return hitidxs;
}
//____________________________________________________________________________________________
std::vector<unsigned int> getModuleIdxsFrompT3(LSTEvent* event, unsigned int pT3) {
  std::vector<unsigned int> hits = getT3HitsFrompT3(event, pT3);
  std::vector<unsigned int> module_idxs;
  auto hitsEvt = event->getHits<HitsExtendedSoA>();
  for (auto& hitIdx : hits) {
    module_idxs.push_back(hitsEvt.moduleIndices()[hitIdx]);
  }
  return module_idxs;
}
//____________________________________________________________________________________________
std::vector<HitType> getHitTypesFrompT3(LSTEvent* event, unsigned int pT3) {
  unsigned int pLS = getpLSFrompT3(event, pT3);
  auto hitTypes = getHitTypesFrompLS(event, pLS);
  hitTypes.insert(hitTypes.end(), Params_T3::kHits, HitType::Phase2OT);
  return hitTypes;
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFrompT3(LSTEvent* event,
                                                                                         unsigned pT3) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFrompT3(event, pT3));
}

// ===============
// ----* pT5 *----
// ===============

//____________________________________________________________________________________________
unsigned int getpLSFrompT5(LSTEvent* event, unsigned int pT5) {
  auto const pixelQuintuplets = event->getPixelQuintuplets();
  auto ranges = event->getRanges();
  auto modulesEvt = event->getModules<ModulesSoA>();
  const unsigned int pLS_offset = ranges.segmentModuleIndices()[modulesEvt.nLowerModules()];
  return pixelQuintuplets.pixelSegmentIndices()[pT5] - pLS_offset;
}

//____________________________________________________________________________________________
unsigned int getT5FrompT5(LSTEvent* event, unsigned int pT5) {
  auto const pixelQuintuplets = event->getPixelQuintuplets();
  return pixelQuintuplets.quintupletIndices()[pT5];
}

//____________________________________________________________________________________________
std::vector<unsigned int> getT3sFrompT5(LSTEvent* event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getT3sFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFrompT5(LSTEvent* event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getLSsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFrompT5(LSTEvent* event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getMDsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getT5HitsFrompT5(LSTEvent* event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getHitsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getpLSHitsFrompT5(LSTEvent* event, unsigned int pT5) {
  unsigned int pLS = getpLSFrompT5(event, pT5);
  return getHitsFrompLS(event, pLS);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFrompT5(LSTEvent* event, unsigned int pT5) {
  auto const& allHits = event->getPixelQuintuplets().hitIndices()[pT5];
  std::vector<unsigned int> hits;
  hits.reserve(allHits.size());
  for (auto const hit : allHits)
    if (hits.empty() || hits.back() != hit)  // should eventually check all and type
      hits.emplace_back(hit);
  return hits;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFrompT5(LSTEvent* event, unsigned int pT5) {
  auto hitsBase = event->getInput<HitsBaseSoA>();
  std::vector<unsigned int> hits = getHitsFrompT5(event, pT5);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsBase.idxs()[hit]);
  return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getModuleIdxsFrompT5(LSTEvent* event, unsigned int pT5) {
  std::vector<unsigned int> hits = getT5HitsFrompT5(event, pT5);
  std::vector<unsigned int> module_idxs;
  auto hitsEvt = event->getHits<HitsExtendedSoA>();
  for (auto& hitIdx : hits) {
    module_idxs.push_back(hitsEvt.moduleIndices()[hitIdx]);
  }
  return module_idxs;
}

//____________________________________________________________________________________________
std::vector<HitType> getHitTypesFrompT5(LSTEvent* event, unsigned int pT5) {
  unsigned int pLS = getpLSFrompT5(event, pT5);
  auto hitTypes = getHitTypesFrompLS(event, pLS);
  hitTypes.insert(hitTypes.end(), Params_T5::kHits, HitType::Phase2OT);
  return hitTypes;
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFrompT5(LSTEvent* event,
                                                                                         unsigned pT5) {
  return convertHitsToHitIdxsAndHitTypes(event, getHitsFrompT5(event, pT5));
}

// ==============
// ----* TC *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromTC(LSTEvent* event, unsigned int iTC) {
  // Get the type of the track candidate
  auto const& trackCandidatesBase = event->getTrackCandidatesBase();
  auto const& trackCandidatesExtended = event->getTrackCandidatesExtended();
  short type = trackCandidatesBase.trackCandidateType()[iTC];
  unsigned int objidx = trackCandidatesExtended.directObjectIndices()[iTC];
  switch (type) {
    case lst::LSTObjType::pT5:
      return getLSsFrompT5(event, objidx);
      break;
    case lst::LSTObjType::pT3:
      return getLSsFrompT3(event, objidx);
      break;
    case lst::LSTObjType::T5:
      return getLSsFromT5(event, objidx);
      break;
    case lst::LSTObjType::pLS:
      return std::vector<unsigned int>();
      break;
    case lst::LSTObjType::T4:
      return getLSsFromT4(event, objidx);
      break;
  }
  throw std::logic_error("Unsupported type " + std::to_string(type));
}

//____________________________________________________________________________________________
std::tuple<std::vector<unsigned int>, std::vector<HitType>> getUnderlyingHitIdxsAndHitTypesFromTC(LSTEvent* event,
                                                                                                  unsigned iTC) {
  // Get the type of the track candidate
  auto const& trackCandidatesBase = event->getTrackCandidatesBase();
  auto const& trackCandidatesExtended = event->getTrackCandidatesExtended();
  short type = trackCandidatesBase.trackCandidateType()[iTC];
  unsigned int objidx = trackCandidatesExtended.directObjectIndices()[iTC];
  switch (type) {
    case lst::LSTObjType::pT5:
      return getHitIdxsAndHitTypesFrompT5(event, objidx);
      break;
    case lst::LSTObjType::pT3:
      return getHitIdxsAndHitTypesFrompT3(event, objidx);
      break;
    case lst::LSTObjType::T5:
      return getHitIdxsAndHitTypesFromT5(event, objidx);
      break;
    case lst::LSTObjType::pLS:
      return getHitIdxsAndHitTypesFrompLS(event, objidx);
      break;
    case lst::LSTObjType::T4:
      return getHitIdxsAndHitTypesFromT4(event, objidx);
      break;
  }
  throw std::logic_error("Unsupported type " + std::to_string(type));
}

std::pair<std::vector<unsigned int>, std::vector<HitType>> getHitIdxsAndHitTypesFromTC(LSTEvent* event,
                                                                                       unsigned int tc_idx) {
  auto const& base = event->getTrackCandidatesBase();
  auto const& ext = event->getTrackCandidatesExtended();
  auto const& hitsBase = event->getInput<HitsBaseSoA>();

  std::vector<unsigned int> hitIdx;
  hitIdx.reserve(Params_TC::kHits);
  std::vector<HitType> hitType;
  hitType.reserve(Params_TC::kHits);

  for (int layerSlot = 0; layerSlot < Params_TC::kLayers; ++layerSlot) {
    if (ext.lowerModuleIndices()[tc_idx][layerSlot] == lst::kTCEmptyLowerModule)
      continue;

    for (unsigned int hitSlot = 0; hitSlot < Params_TC::kHitsPerLayer; ++hitSlot) {
      const unsigned int hitLocal = base.hitIndices()[tc_idx][layerSlot][hitSlot];

      if (hitLocal == lst::kTCEmptyHitIdx)
        continue;

      // Get the GLOBAL ntuple indices
      const auto hitGlobal = hitsBase.idxs()[hitLocal];

      // Determine the type from the hit's detid
      const auto type = (hitsBase.detid()[hitLocal] == kPixelModuleId) ? HitType::Pixel : HitType::Phase2OT;

      // Push the GLOBAL index and type
      hitIdx.push_back(hitGlobal);
      hitType.push_back(type);
    }
  }
  return {hitIdx, hitType};
}
