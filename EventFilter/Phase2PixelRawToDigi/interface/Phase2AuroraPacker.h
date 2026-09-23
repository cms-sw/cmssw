#ifndef EventFilter_Phase2PixelRawToDigi_Phase2AuroraPacker_H
#define EventFilter_Phase2PixelRawToDigi_Phase2AuroraPacker_H

// RD53B Aurora-format helpers used by BitStreamToAuroraProducer to compute per-chip Aurora wire-size.
// Three layers
// - 1 : Chip bitstream level
// - 2 : Aurora framing
// - 3 : Aurora Service block

#include <vector>
#include <algorithm>
#include <cstddef>

#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"

namespace Phase2AuroraPacker {

  using namespace Phase2DAQFormatSpecification;

  // Convert integer to binary
  inline std::vector<bool> intToBinary(int value, int length) {
    std::vector<bool> bin;
    for (int i = 0; value > 0; i++) {
      if (i >= length)
        break;
      bin.push_back(bool(value % 2));
      value = value / 2;
    }
    while ((int)bin.size() < length)
      bin.push_back(false);
    std::reverse(bin.begin(), bin.end());
    return bin;
  }

  // Per-event tag: 8 bits for the first event in a stream group, 11 bits for every subsequent events
  inline std::vector<bool> makeEventTag(int event, bool isFirstInStream) {
    return intToBinary(event, isFirstInStream ? 8 : 11);
  }

  // Aurora data-block framing.
  //   chipId == -1 : single-chip mode, 1-b NS + 63-b payload per block.
  //   chipId 0..3  : multi-chip mode,  1-b NS + 2-b ID + 61-b payload per block.
  // FIXME NS (End-Stream flag) needs to be appended to the last bit (for now ignore the effect as the size won't matter from this)
  inline std::vector<bool> applyBlocking(const std::vector<bool>& stream, int chipId) {
    if (chipId < -1 || chipId > 3)
      throw cms::Exception("Phase2AuroraPacker") << "applyBlocking: chipId must be in [-1, 3], got " << chipId;

    std::vector<bool> newStream;
    const int streamSize = stream.size();

    std::vector<bool> binaryChipId = intToBinary(chipId, AURORA_CHIP_ID_BITS);

    const int headerBits = AURORA_NS_BITS + (chipId >= 0 ? AURORA_CHIP_ID_BITS : 0);
    const int step = AURORA_BLOCK_BODY_BITS - headerBits;
    const int nBlocks = (streamSize + step - 1) / step;
    newStream.reserve(streamSize + nBlocks * headerBits);

    for (int index = 0; index < nBlocks; ++index) {
      newStream.push_back(newStream.empty());
      if (chipId >= 0) {
        newStream.insert(newStream.end(), binaryChipId.begin(), binaryChipId.end());
      }
      int offsetLow = index * step;
      int offsetHigh = std::min((index + 1) * step, streamSize);
      newStream.insert(newStream.end(), stream.begin() + offsetLow, stream.begin() + offsetHigh);
    }

    return newStream;
  }

  // Orphan padding: zero-pad the trailing partial block-body to AURORA_BLOCK_BODY_BITS.
  inline std::vector<bool> orphanPad(const std::vector<bool>& stream) {
    std::vector<bool> newStream(stream);
    newStream.reserve((newStream.size() / AURORA_BLOCK_BODY_BITS + 1) * AURORA_BLOCK_BODY_BITS);
    while (newStream.size() % AURORA_BLOCK_BODY_BITS > 0)
      newStream.push_back(false);
    return newStream;
  }

  // End-of-Stream marker: Append a full block-body of zeros if the orphan tail is shorter than the spec minimum.
  inline std::vector<bool> applyEosMarker(const std::vector<bool>& paddedStream) {
    int trailingZeros = 0;
    for (auto it = paddedStream.rbegin(); it != paddedStream.rend(); ++it) {
      if (!*it) {
        if (++trailingZeros >= AURORA_EOS_MIN_TRAILING_ZEROS)
          break;
      } else {
        break;
      }
    }
    if (trailingZeros >= AURORA_EOS_MIN_TRAILING_ZEROS)
      return paddedStream;

    std::vector<bool> result;
    result.reserve(paddedStream.size() + AURORA_BLOCK_BODY_BITS);
    result.insert(result.end(), paddedStream.begin(), paddedStream.end());
    for (int i = 0; i < AURORA_BLOCK_BODY_BITS; ++i)
      result.push_back(false);
    return result;
  }

  // Aurora 64b/66b sync header: prepend AURORA_SYNC_HEADER_BITS to every block body.
  inline std::vector<bool> applyHeaderBlocks(const std::vector<bool>& blockStream) {
    if (blockStream.size() % AURORA_BLOCK_BODY_BITS != 0)
      throw cms::Exception("Phase2AuroraPacker")
          << "applyHeaderBlocks: input size (" << blockStream.size()
          << " bits) is not a multiple of AURORA_BLOCK_BODY_BITS (" << AURORA_BLOCK_BODY_BITS << ")";
    int nBlocks = blockStream.size() / AURORA_BLOCK_BODY_BITS;
    std::vector<bool> result;
    result.reserve(nBlocks * AURORA_BLOCK_TOTAL_BITS);
    for (int b = 0; b < nBlocks; ++b) {
      for (int i = 0; i < AURORA_SYNC_HEADER_BITS; ++i)
        result.push_back(false);
      result.insert(result.end(),
                    blockStream.begin() + b * AURORA_BLOCK_BODY_BITS,
                    blockStream.begin() + (b + 1) * AURORA_BLOCK_BODY_BITS);
    }
    return result;
  }

  // Aurora service block: insert one full 66-b K-block per nD data blocks
  inline std::vector<bool> applyServiceBlocks(const std::vector<bool>& headeredStream,
                                                int nD = AURORA_SERVICE_BLOCK_INTERVAL_DEFAULT) {
    if (nD < AURORA_SERVICE_BLOCK_INTERVAL_MIN || nD > AURORA_SERVICE_BLOCK_INTERVAL_MAX)
      throw cms::Exception("Phase2AuroraPacker")
          << "applyServiceBlocks: nD=" << nD << " out of range [" << AURORA_SERVICE_BLOCK_INTERVAL_MIN << ", "
          << AURORA_SERVICE_BLOCK_INTERVAL_MAX << "]";
    if (headeredStream.size() % AURORA_BLOCK_TOTAL_BITS != 0)
      throw cms::Exception("Phase2AuroraPacker")
          << "applyServiceBlocks: input size (" << headeredStream.size()
          << " bits) is not a multiple of AURORA_BLOCK_TOTAL_BITS (" << AURORA_BLOCK_TOTAL_BITS << ")";
    int nDataBlocks = headeredStream.size() / AURORA_BLOCK_TOTAL_BITS;
    int nServiceBlocks = nDataBlocks / nD;

    std::vector<bool> result;
    result.reserve((nDataBlocks + nServiceBlocks) * AURORA_BLOCK_TOTAL_BITS);
    int dataCount = 0;
    for (int b = 0; b < nDataBlocks; ++b) {
      result.insert(result.end(),
                    headeredStream.begin() + b * AURORA_BLOCK_TOTAL_BITS,
                    headeredStream.begin() + (b + 1) * AURORA_BLOCK_TOTAL_BITS);
      if (++dataCount == nD) {
        for (int i = 0; i < AURORA_BLOCK_TOTAL_BITS; ++i)
          result.push_back(false);
        dataCount = 0;
      }
    }
    return result;
  }

}  // namespace Phase2AuroraPacker

#endif  // EventFilter_Phase2PixelRawToDigi_Phase2AuroraPacker_H
