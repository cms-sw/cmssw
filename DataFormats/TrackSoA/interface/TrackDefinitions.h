#ifndef DataFormats_Track_interface_TrackDefinitions_h
#define DataFormats_Track_interface_TrackDefinitions_h
#include <string>
#include <algorithm>
#include <stdexcept>

namespace pixelTrack {

  enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity, notQuality };
  constexpr uint32_t qualitySize{uint8_t(Quality::notQuality)};
  constexpr std::string_view qualityName[qualitySize]{"bad", "edup", "dup", "loose", "strict", "tight", "highPurity"};
  inline Quality qualityByName(std::string_view name) {
    auto qp = std::find(qualityName, qualityName + qualitySize, name) - qualityName;
    auto ret = static_cast<Quality>(qp);

    if (ret == pixelTrack::Quality::notQuality)
      throw std::invalid_argument(std::string(name) + " is not a pixelTrack::Quality!");

    return ret;
  }

#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumber() { return 2 * 1024; }
#else
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumber() { return 32 * 1024; }
#endif

}  // namespace pixelTrack

#endif
