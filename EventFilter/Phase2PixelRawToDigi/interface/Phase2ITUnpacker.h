#ifndef EventFilter_Phase2PixelRawToDigi_Phase2ITUnpacker_h
#define EventFilter_Phase2PixelRawToDigi_Phase2ITUnpacker_h

// Shared functions for the IT unpacking chain

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITBitReader.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChip.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITQCore.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace Phase2ITUnpacker {

  inline uint32_t readWord(const unsigned char* dataPtr, int wordIdx) {
    const int byteIdx = wordIdx * 4;
    return (static_cast<uint32_t>(dataPtr[byteIdx]) << 24) | (static_cast<uint32_t>(dataPtr[byteIdx + 1]) << 16) |
           (static_cast<uint32_t>(dataPtr[byteIdx + 2]) << 8) | static_cast<uint32_t>(dataPtr[byteIdx + 3]);
  }

  // FIXME this later need to be properly treated in simulation
  // KEEP shifts each chip boundary to the gap midline - only for in & out validation purposes
  // DROP/AGGREGATE keep the physical extents, either removing the ToT or stacking up the ToT
  inline bool parseKeepMode(const std::string& s, const char* who) {
    if (s == "DROP" || s == "AGGREGATE")
      return false;
    if (s == "KEEP")
      return true;
    throw cms::Exception(who) << "handleGapPixels must be one of DROP/KEEP/AGGREGATE, got '" << s << "'";
  }

  // Validate the SLinkRocket wrapper and return the IT body it wraps
  inline const unsigned char* stripSLinkWrapper(const unsigned char* fragPtr,
                                                uint32_t fragSize,
                                                int fedId,
                                                int& fedSizeInWords) {
    using namespace Phase2DAQFormatSpecification;
    constexpr unsigned int kSlinkHdrBytes = sizeof(SLinkRocketHeader_v3);
    constexpr unsigned int kSlinkTrlBytes = sizeof(SLinkRocketTrailer_v3);
    auto const* sh = reinterpret_cast<const SLinkRocketHeader_v3*>(fragPtr);
    auto const* st = reinterpret_cast<const SLinkRocketTrailer_v3*>(fragPtr + fragSize - kSlinkTrlBytes);
    if (!sh->verifyMarker())
      throw cms::Exception("Phase2ITUnpacker") << "Invalid SLinkRocket BOE for fed " << fedId;
    if (!st->verifyMarker())
      throw cms::Exception("Phase2ITUnpacker") << "Invalid SLinkRocket EOE for fed " << fedId;
    if (st->eventLenBytes() != fragSize)
      throw cms::Exception("Phase2ITUnpacker") << "SLinkRocket trailer length mismatch for fed " << fedId
                                               << ": trailer says " << st->eventLenBytes()
					       << ", actual " << fragSize;
    fedSizeInWords = static_cast<int>(fragSize / BYTES_PER_WORD) - (kSlinkHdrBytes + kSlinkTrlBytes) / BYTES_PER_WORD;
    return fragPtr + kSlinkHdrBytes;
  }

  // FIXME for now this works because we are assuming 4 lines of 0xFFFFFFFF for
  // both headers and trailers. Later we have to come up with something more
  // concrete to parse out these 4 lines.
  inline bool verifyHeaderTrailerPattern(const unsigned char* dataPtr, int wordIdx) {
    using namespace Phase2DAQFormatSpecification;
    for (int i = 0; i < HEADER_TRAILER_LINES; i++) {
      if (readWord(dataPtr, wordIdx + i) != HEADER_TRAILER_PATTERN)
        return false;
    }
    return true;
  }

  inline int findTrailerStart(const unsigned char* dataPtr, int fedSizeInWords) {
    using namespace Phase2DAQFormatSpecification;
    // Start searching from the end, going backwards
    for (int i = fedSizeInWords - HEADER_TRAILER_LINES; i >= HEADER_TRAILER_LINES; --i) {
      if (verifyHeaderTrailerPattern(dataPtr, i))
        return i;
    }
    return -1;  // trailer not found
  }

  // Word span [start, end) of one module inside a FED body.
  struct ModuleSpan {
    int start;
    int end;
  };

  // Walk through the modules of an FED
  template <typename ModuleF>
  void forEachModule(
      const unsigned char* dataPtr, int fedSizeInWords, int trailerStart, int numModules, ModuleF&& module) {
    using namespace Phase2DAQFormatSpecification;
    const int offsetStart = HEADER_TRAILER_LINES;

    // The offset block is padded to a 128-bit boundary at its end
    const int offsetBits = numModules * BITS_PER_WORD;
    const int paddingBits = (BITS_PER_CHUNK - (offsetBits % BITS_PER_CHUNK)) % BITS_PER_CHUNK;
    const int dataBlockStart = offsetStart + numModules + paddingBits / BITS_PER_WORD;

    for (int modIdx = 0; modIdx < numModules; modIdx++) {
      const int moduleStartWord = dataBlockStart + static_cast<int>(readWord(dataPtr, offsetStart + modIdx));
      if (moduleStartWord < 0 || moduleStartWord >= fedSizeInWords) {
        edm::LogWarning("Phase2ITUnpacker")
            << "Module offset out of FED bounds: module index " << modIdx << " moduleStartWord=" << moduleStartWord
            << " fedSize=" << fedSizeInWords << ". Skipping module.";
        continue;
      }
      // End of this module's data = start of the next module (or the trailer for the last)
      int moduleEndWord = (modIdx + 1 < numModules)
                              ? (dataBlockStart + static_cast<int>(readWord(dataPtr, offsetStart + modIdx + 1)))
                              : trailerStart;
      if (moduleEndWord > fedSizeInWords)
        moduleEndWord = fedSizeInWords;
      module(modIdx, ModuleSpan{moduleStartWord, moduleEndWord});
    }
  }

  // Walk through the chips of a module
  // nBits is kept at 0 for a malformed payload.
  template <typename ChipF>
  void forEachChip(const unsigned char* dataPtr, ModuleSpan span, int fedSizeInWords, ChipF&& chip) {
    using namespace Phase2DAQFormatSpecification;
    int chipCursor = span.start;
    for (int chipId = 0; chipCursor < span.end; chipId++) {
      const uint32_t chipHeader = readWord(dataPtr, chipCursor);
      const uint32_t magic = (chipHeader >> 28) & 0xF;
      // FIXME ignoring the chip error bits with Dummy for now
      const uint32_t endBit = (chipHeader >> 16) & 0x1F;
      const uint32_t sizeWords = chipHeader & 0xFFFF;

      if (magic != CHIP_HEADER_MAGIC)
        break;

      // Create the stream length in bits.
      //   endBit == 0  -> last word is full or chip is empty: size = sizeWords * 32
      //   endBit  > 0  -> last word holds endBit real bits:   size = (sizeWords - 1) * 32 + endBit
      const int bitstreamSize = (endBit == 0) ? static_cast<int>(sizeWords * BITS_PER_WORD)
                                              : static_cast<int>((sizeWords - 1) * BITS_PER_WORD + endBit);

      uint32_t nBits = 0;
      if (bitstreamSize > 0) {
        const int fullWords = bitstreamSize / BITS_PER_WORD;
        const int wordsNeeded = fullWords + (bitstreamSize % BITS_PER_WORD > 0 ? 1 : 0);
        if (chipCursor + 1 + wordsNeeded > fedSizeInWords) {
          edm::LogWarning("Phase2ITUnpacker")
              << "Bitstream read out of FED bounds: startWord=" << chipCursor + 1 << ", needs " << wordsNeeded
              << " words, FED size " << fedSizeInWords << " words. Treating chip as empty.";
        } else {
          nBits = static_cast<uint32_t>(bitstreamSize);
        }
      }
      chip(chipId, chipCursor + 1, nBits);
      chipCursor += 1 + static_cast<int>(sizeWords);
    }
  }

  // Decode chip bit stream
  inline void decodeChip(
      Phase2ITBitReader& reader, int chipId, int subtype, bool dropTot, bool keepMode, edm::DetSet<PixelDigi>& detSet) {
    using namespace Phase2DAQFormatSpecification;
    int currentCol = 0;
    int currentRow = 0;
    int previousRow = -1;
    bool previousIsLast = true;

    while (!reader.atEnd()) {
      // Always start with a fresh ccol only at the start of a new column group
      if (previousIsLast) {
        currentCol = reader.bits(6);
      }

      const bool islast = reader.next();
      const bool isneighbor = reader.next();

      // isneighbor=1 means the previous qrow address is current_qrow - 1, so
      // the qrow field is omitted and we give previous + 1.
      if (isneighbor) {
        currentRow = previousRow + 1;
      } else {
        currentRow = reader.bits(8);
      }

      std::array<bool, 16> hitmap = Phase2ITQCore::decodeHitmap(reader);
      int numHits = std::count(hitmap.begin(), hitmap.end(), true);
      // In dropTot mode the encoder skipped the ToT bits: emit adc=0 for every
      // hit, otherwise read 4 bits per hit as the ToT/ADC value.
      std::array<int, 16> adcValues{};
      if (!dropTot)
        adcValues = Phase2ITQCore::decodeADCs(reader, numHits);

      int adcIndex = 0;
      for (int i = 0; i < HITMAP_SIZE; i++) {
        if (!hitmap[i])
          continue;
        int rocRow = i / 8;
        int rocCol = i % 8;
        int shiftedRow = rocRow * 2 + (rocCol % 2);
        int shiftedCol = rocCol / 2;
        int shiftedIndex = shiftedRow * HITMAP_COL + shiftedCol;
        auto [localRow, localCol] = Phase2ITChip::decodeQCoreIndex(shiftedIndex);
        auto [globalRow, globalCol] = Phase2ITChip::getGlobalPixelCoordinate(
            chipId, subtype, currentCol, currentRow, localCol, localRow, keepMode);
        detSet.push_back(PixelDigi(globalRow, globalCol, adcValues[adcIndex++]));
      }

      previousIsLast = islast;
      previousRow = currentRow;
    }
  }

}  // namespace Phase2ITUnpacker

#endif  // EventFilter_Phase2PixelRawToDigi_Phase2ITUnpacker_h
