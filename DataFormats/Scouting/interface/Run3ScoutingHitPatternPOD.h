#ifndef DataFormats_Scouting_Run3Scouting_HitPatternPOD_h
#define DataFormats_Scouting_Run3Scouting_HitPatternPOD_h

#include <vector>

struct Run3ScoutingHitPatternPOD {
  uint8_t hitCount;
  uint8_t beginTrackHits;
  uint8_t endTrackHits;
  uint8_t beginInner;
  uint8_t endInner;
  uint8_t beginOuter;
  uint8_t endOuter;
  std::vector<uint16_t> hitPattern;
};

#endif
