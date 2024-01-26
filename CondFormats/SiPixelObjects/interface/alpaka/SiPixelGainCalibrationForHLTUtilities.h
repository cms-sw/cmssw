#ifndef CondFormats_SiPixelObjects_interface_alpaka_SiPixelGainCalibrationForHLTUtilities_h
#define CondFormats_SiPixelObjects_interface_alpaka_SiPixelGainCalibrationForHLTUtilities_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"

struct SiPixelGainUtilities {
  ALPAKA_FN_HOST_ACC ALPAKA_FN_ACC ALPAKA_FN_INLINE static std::pair<float, float> getPedAndGain(
      const SiPixelGainCalibrationForHLTSoAConstView& view,
      uint32_t moduleInd,
      int col,
      int row,
      bool& isDeadColumn,
      bool& isNoisyColumn) {
    auto start = view.modStarts()[moduleInd];
    auto end = view.modEnds()[moduleInd];
    auto nCols = view.modCols()[moduleInd];
    // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
    unsigned int lengthOfColumnData = (end - start) / nCols;
    unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block
    unsigned int numberOfDataBlocksToSkip = row / view.numberOfRowsAveragedOver();

    auto offset = start + col * lengthOfColumnData + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip;
    assert(offset < end);
    assert(offset < 3088384);
    assert(0 == offset % 2);

    auto lp = view.v_pedestals();
    auto s = lp[offset / 2];

    isDeadColumn = (s.ped & 0xFF) == view.deadFlag();
    isNoisyColumn = (s.ped & 0xFF) == view.noisyFlag();
    float decodeGain = float(s.gain & 0xFF) * view.gainPrecision() + view.minGain();
    float decodePed = float(s.ped & 0xFF) * view.pedPrecision() + view.minPed();

    return std::make_pair(decodePed, decodeGain);
  };
};

#endif  //CondFormats_SiPixelObjects_interface_alpaka_SiPixelGainCalibrationForHLTUtilities_h