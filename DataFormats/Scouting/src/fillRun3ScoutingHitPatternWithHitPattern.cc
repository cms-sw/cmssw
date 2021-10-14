#include "DataFormats/Scouting/interface/Run3ScoutingHitPattern.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"

/* This file contains the function used to read HitPattern and fill Run3ScoutingHitPattern (= HitPattern, v13) */

constexpr unsigned short HitSize = reco::HitPattern::HIT_LENGTH;
constexpr unsigned short PatternSize = reco::HitPattern::ARRAY_LENGTH;
constexpr int MaxHits = reco::HitPattern::MaxHits;

auto getHitFromTrackRecoHitPattern(const uint16_t hitPattern[], const int position, const int hitCount) {
  if UNLIKELY ((position < 0 || position >= hitCount)) {
    return Run3ScoutingHitPattern::EMPTY_PATTERN;
  }

  const uint16_t bitEndOffset = (position + 1) * HitSize;
  const uint8_t secondWord = (bitEndOffset >> 4);
  const uint8_t secondWordBits = bitEndOffset & (16 - 1);

  if (secondWordBits >= HitSize) {
    // full block is in this word
    const uint8_t lowBitsToTrash = secondWordBits - HitSize;
    uint16_t myResult = (hitPattern[secondWord] >> lowBitsToTrash) & ((1 << HitSize) - 1);
    return myResult;
  } else {
    const uint8_t firstWordBits = HitSize - secondWordBits;
    const uint16_t firstWordBlock = hitPattern[secondWord - 1] >> (16 - firstWordBits);
    const uint16_t secondWordBlock = hitPattern[secondWord] & ((1 << secondWordBits) - 1);
    uint16_t myResult = firstWordBlock + (secondWordBlock << firstWordBits);
    return myResult;
  }
}

auto hitTypeFromTrackRecoHitPattern(const uint16_t pattern,
                                    const unsigned short HitTypeOffset,
                                    const unsigned short HitTypeMask) {
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
}

void Run3ScoutingHitPattern::fillRun3ScoutingHitPatternWithHitPattern(const reco::HitPattern trackRecoHitPattern) {
  for (int i = 0; i < MaxHits; i++) {
    if (i >= trackRecoHitPattern.hitCount)
      break;

    uint16_t pattern = getHitFromTrackRecoHitPattern(trackRecoHitPattern.hitPattern, i, trackRecoHitPattern.hitCount);
    if (pattern == 0) {
      break;
    }
    if (!this->appendHit(
            pattern,
            hitTypeFromTrackRecoHitPattern(pattern, reco::HitPattern::HitTypeOffset, reco::HitPattern::HitTypeMask))) {
      break;
    }
  }

  this->hitCount = trackRecoHitPattern.hitCount;
  this->beginTrackHits = trackRecoHitPattern.beginTrackHits;
  this->endTrackHits = trackRecoHitPattern.endTrackHits;
  this->beginInner = trackRecoHitPattern.beginInner;
  this->endInner = trackRecoHitPattern.endInner;
  this->beginOuter = trackRecoHitPattern.beginOuter;
  this->endOuter = trackRecoHitPattern.endOuter;
}
