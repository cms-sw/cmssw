#ifndef DataFormats_Scouting_Run3ScoutingHitPatternPOD_h
#define DataFormats_Scouting_Run3ScoutingHitPatternPOD_h

#include <vector>
#include <cstdint>

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
