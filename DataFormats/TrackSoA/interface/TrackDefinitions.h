#ifndef DataFormats_TrackSoA_interface_TrackDefinitions_h
#define DataFormats_TrackSoA_interface_TrackDefinitions_h
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

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

  enum class Iteration : uint8_t {
    promptHighPt,
    promptLowPt,
    notIteration
  };  // Not sure if a notIteration will be needed
  constexpr uint32_t iterationSize{uint8_t(Iteration::notIteration)};
  constexpr std::string_view iterationName[iterationSize]{"promptHighPt", "promptLowPt"};
  inline Iteration iterationByName(std::string_view name) {
    auto qp = std::find(iterationName, iterationName + iterationSize, name) - iterationName;
    auto ret = static_cast<Iteration>(qp);

    if (ret == pixelTrack::Iteration::notIteration)
      throw std::invalid_argument(std::string(name) + " is not a pixelTrack::Iteration!");

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
