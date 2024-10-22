#include "DataFormats/TrackReco/interface/HitPattern.h"

/* This file contains the function used to read back v12 versions of HitPatterns from a ROOT file.
   The function is called by a ROOT IO rule.
 */

using namespace reco;

namespace {
  constexpr unsigned short HitSize = 11;
  constexpr unsigned short PatternSize = 50;
  constexpr int MaxHitsV12 = (PatternSize * sizeof(uint16_t) * 8) / HitSize;

  auto getHitFromOldHitPattern(const uint16_t hitPattern[], const int position) {
    const uint16_t bitEndOffset = (position + 1) * HitSize;
    const uint8_t secondWord = (bitEndOffset >> 4);
    const uint8_t secondWordBits = bitEndOffset & (16 - 1);  // that is, bitEndOffset % 32
    if (secondWordBits >= HitSize) {
      // full block is in this word
      const uint8_t lowBitsToTrash = secondWordBits - HitSize;
      return (hitPattern[secondWord] >> lowBitsToTrash) & ((1 << HitSize) - 1);
    }
    const uint8_t firstWordBits = HitSize - secondWordBits;
    const uint16_t firstWordBlock = hitPattern[secondWord - 1] >> (16 - firstWordBits);
    const uint16_t secondWordBlock = hitPattern[secondWord] & ((1 << secondWordBits) - 1);
    return firstWordBlock + (secondWordBlock << firstWordBits);
  }

  auto hitTypeFromOldHitPattern(const uint16_t pattern) {
    // for this version we just have to add a 0 bit to the top of the pattern
    constexpr unsigned short HitTypeMask = 0x3;
    constexpr unsigned short HitTypeOffset = 0;

    constexpr uint16_t VALID_CONST = (uint16_t)TrackingRecHit::valid;
    constexpr uint16_t MISSING_CONST = (uint16_t)TrackingRecHit::missing;
    constexpr uint16_t INACTIVE_CONST = (uint16_t)TrackingRecHit::inactive;
    constexpr uint16_t BAD_CONST = (uint16_t)TrackingRecHit::bad;

    const uint16_t rawHitType = (pattern >> HitTypeOffset) & HitTypeMask;

    TrackingRecHit::Type hitType = TrackingRecHit::valid;
    switch (rawHitType) {
      case VALID_CONST:
        hitType = TrackingRecHit::valid;
        break;
      case MISSING_CONST:
        hitType = TrackingRecHit::missing;
        break;
      case INACTIVE_CONST:
        hitType = TrackingRecHit::inactive;
        break;
      case BAD_CONST:
        hitType = TrackingRecHit::bad;
        break;
    }
    return hitType;
  };
}  // namespace

bool reco::HitPattern::fillNewHitPatternWithOldHitPattern_v12(const uint16_t oldHitPattern[],
                                                              uint8_t hitCount,
                                                              uint8_t beginTrackHits,
                                                              uint8_t endTrackHits,
                                                              uint8_t beginInner,
                                                              uint8_t endInner,
                                                              uint8_t beginOuter,
                                                              uint8_t endOuter,
                                                              HitPattern* newObj) {
  newObj->clear();
  bool ret = true;
  for (int i = 0; i < MaxHitsV12; i++) {
    uint16_t pattern = getHitFromOldHitPattern(oldHitPattern, i);
    if (pattern == 0) {
      break;
    }
    if (!newObj->appendHit(pattern, hitTypeFromOldHitPattern(pattern))) {
      ret = false;
      break;
    }
  }
  newObj->hitCount = hitCount;
  newObj->beginTrackHits = beginTrackHits;
  newObj->endTrackHits = endTrackHits;
  newObj->beginInner = beginInner;
  newObj->endInner = endInner;
  newObj->beginOuter = beginOuter;
  newObj->endOuter = endOuter;
  return ret;
}
