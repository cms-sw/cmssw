#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT.h"

#include <iostream>
#include <cassert>

int main() {
  // test quality

  auto q = pixelTrack::qualityByName("tight");
  assert(pixelTrack::Quality::tight == q);
  q = pixelTrack::qualityByName("toght");
  assert(pixelTrack::Quality::notQuality == q);

  for (uint32_t i = 0; i < pixelTrack::qualitySize; ++i) {
    auto const qt = static_cast<pixelTrack::Quality>(i);
    auto q = pixelTrack::qualityByName(pixelTrack::qualityName[i]);
    assert(qt == q);
  }

  return 0;
}
