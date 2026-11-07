#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

struct SiPixelMorphingConfig {
  bool applyDigiMorphing;
  uint32_t maxFakesInModule;
  uint32_t numMorphingModules;
};

namespace pixelClustering {

  namespace phase2Morphing {
    // Phase-2 pixel modules are too large for a whole-module bitmap in shared memory (a quad module has
    // 1354 x 434 pixels, 145 kB for the two 1-bit buffers), so the digi morphing uses the same 2-buffer
    // scheme as Phase-1, applied to one strip of `stripCols` columns at a time.
    //
    // Each word holds 32 consecutive rows (x) of one column (y), as in Phase-1.
    // The buffers cover the strip plus `halo` columns on each side: the erosion of the strip edge column needs
    // the dilated value of the next column, which in turn depends on the column after that.
    constexpr uint32_t valuesPerWord = sizeof(uint32_t) * 8;  // 32 values per 32-bit word
    constexpr uint32_t maxRows = 1376;  // rows (x) in a quad module (1354 with the big pixels), rounded up to a word
    constexpr uint32_t maxWordsPerColumn = maxRows / valuesPerWord;  // 43 words per column
    constexpr uint32_t stripCols = 44;                               // columns (y) processed per strip
    constexpr uint32_t halo = 2;                                     // extra columns on each side
    constexpr uint32_t bufferCols = stripCols + 2 * halo;            // 48 columns
    constexpr uint32_t size = bufferCols * maxWordsPerColumn;        // 2064 32-bit words, 8 kB per buffer
    static_assert(maxRows % valuesPerWord == 0);

    // default maximum number of pixels recovered per module; Phase-2 modules have about 250 pixels on average
    // at PU 200, and very rarely more than 1000
    constexpr uint32_t defaultMaxFakesInModule = 1000;

    // the size of each morphing module is passed to the device as (rows << 16 | columns)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t packSize(uint16_t rows, uint16_t cols) {
      return (static_cast<uint32_t>(rows) << 16) | cols;
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t rows(uint32_t packed) { return packed >> 16; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t cols(uint32_t packed) { return packed & 0xffff; }
  }  // namespace phase2Morphing

}  // namespace pixelClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h
