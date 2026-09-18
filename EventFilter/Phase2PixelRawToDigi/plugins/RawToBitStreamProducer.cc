// EDProducer that takes RawDataBuffer and produces ITChipBitStream
// Very first step of unpacker

#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChipBitStream.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/SLinkModuleMap.h"

using namespace Phase2DAQFormatSpecification;

class RawToBitStreamProducer : public edm::stream::EDProducer<> {
public:
  explicit RawToBitStreamProducer(const edm::ParameterSet&);
  ~RawToBitStreamProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  uint32_t readWord(const unsigned char* dataPtr, int wordIdx) const;
  std::string wordToHexString(uint32_t word) const;

  // Debugging functions for helper methods
  std::string getBitString(const std::vector<bool>& bits, size_t start, size_t len) const;
  void dumpBitStream(const std::vector<bool>& bits, size_t position) const;

  bool verifyHeaderTrailerPattern(const unsigned char* dataPtr, int wordIdx) const;
  int findTrailerStart(const unsigned char* dataPtr, int fedSizeInWords) const;
  std::vector<bool> extractBitStream(const unsigned char* dataPtr,
                                     int startWord,
                                     int bitstreamSize,
                                     int fedSizeInWords) const;

  void processFED(const unsigned char* dataPtr,
                  int fedSizeInWords,
                  int fedId,
                  edmNew::DetSetVector<Phase2ITChipBitStream>& output);

  const edm::EDGetTokenT<RawDataBuffer> fedRawDataToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;

  std::unique_ptr<SLinkModuleMap> slinkMap_;

  bool debug_ = false;
};

RawToBitStreamProducer::RawToBitStreamProducer(const edm::ParameterSet& iConfig)
    : fedRawDataToken_(consumes<RawDataBuffer>(iConfig.getParameter<edm::InputTag>("fedRawDataCollection"))),
      cablingMapToken_(
          esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      debug_(iConfig.getUntrackedParameter<bool>("debug", false)) {
  produces<edmNew::DetSetVector<Phase2ITChipBitStream>>();
}

void RawToBitStreamProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("fedRawDataCollection", edm::InputTag("rawDataCollector"));
  desc.addUntracked<bool>("debug", false);
  descriptions.add("phase2RawToBitStreamProducer", desc);
}

void RawToBitStreamProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  slinkMap_ = std::make_unique<SLinkModuleMap>(iSetup.getData(cablingMapToken_));
}

void RawToBitStreamProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto output = std::make_unique<edmNew::DetSetVector<Phase2ITChipBitStream>>();
  edm::Handle<RawDataBuffer> rawBuf;
  iEvent.getByToken(fedRawDataToken_, rawBuf);
  if (!rawBuf.isValid()) {
    throw cms::Exception("RawToBitStreamProducer") << "Invalid RawDataBuffer";
  }

  const unsigned int SLINK_HDR_BYTES = sizeof(SLinkRocketHeader_v3);
  const unsigned int SLINK_TRL_BYTES = sizeof(SLinkRocketTrailer_v3);
  const unsigned int WRAPPER_WORDS   = (SLINK_HDR_BYTES + SLINK_TRL_BYTES) / BYTES_PER_WORD;

  for (const auto& entry : slinkMap_->fedIdToDetIds()) {
    int fedId = entry.first;
    auto frag = rawBuf->fragmentData(static_cast<uint32_t>(fedId));
    if (!frag.isValid()) {
      throw cms::Exception("RawToBitStreamProducer")
          << "Missing RawDataBuffer fragment for fed " << fedId
          << ": cabling map lists this FED but the buffer has no source for it.";
    }
    auto span = frag.data();
    const unsigned char* fragPtr = span.data();
    uint32_t fragSize            = span.size();

    // Validate SLinkRocket wrapper.
    auto const* sh = reinterpret_cast<const SLinkRocketHeader_v3*>(fragPtr);
    auto const* st = reinterpret_cast<const SLinkRocketTrailer_v3*>(fragPtr + fragSize - SLINK_TRL_BYTES);
    if (!sh->verifyMarker()) {
      throw cms::Exception("RawToBitStreamProducer")
          << "Invalid SLinkRocket BOE for fed " << fedId;
    }
    if (!st->verifyMarker()) {
      throw cms::Exception("RawToBitStreamProducer")
          << "Invalid SLinkRocket EOE for fed " << fedId;
    }
    if (st->eventLenBytes() != fragSize) {
      throw cms::Exception("RawToBitStreamProducer")
          << "SLinkRocket trailer length mismatch for fed " << fedId
          << ": trailer says " << st->eventLenBytes() << ", actual " << fragSize;
    }

    // Hand the IT-specific body (IT header + offsets + data + IT trailer)
    const unsigned char* dataPtr = fragPtr + SLINK_HDR_BYTES;
    int fedSizeInWords = static_cast<int>(fragSize / BYTES_PER_WORD) - WRAPPER_WORDS;
    processFED(dataPtr, fedSizeInWords, fedId, *output);
  }
  iEvent.put(std::move(output));
}

void RawToBitStreamProducer::processFED(const unsigned char* dataPtr,
                                        int fedSizeInWords,
                                        int fedId,
                                        edmNew::DetSetVector<Phase2ITChipBitStream>& output) {
  const std::vector<uint32_t>& detIds = slinkMap_->detIdsForFedId(fedId);
  if (!verifyHeaderTrailerPattern(dataPtr, 0)) {
    throw cms::Exception("RawToBitStreamProducer") << "Invalid header in FEDRawData";
  }
  int trailerStart = findTrailerStart(dataPtr, fedSizeInWords);
  if (trailerStart < 0) {
    throw cms::Exception("RawToBitStreamProducer") << "Invalid trailer in FEDRawData";
  }

  int numModules = detIds.size();
  int offsetStart = HEADER_TRAILER_LINES;

  // Block 2: read N module offsets (one 32-bit word per module)
  std::vector<uint32_t> moduleOffsets;
  moduleOffsets.reserve(numModules);
  for (int i = 0; i < numModules; i++) {
    moduleOffsets.push_back(readWord(dataPtr, offsetStart + i));
  }

  // The offset block is padded to a 128-bit boundary at its end.
  int offsetBits = numModules * BITS_PER_WORD;
  int paddingBits = (BITS_PER_CHUNK - (offsetBits % BITS_PER_CHUNK)) % BITS_PER_CHUNK;
  int paddingWords = paddingBits / BITS_PER_WORD;
  int dataBlockStart = offsetStart + numModules + paddingWords;

  // Block 3: walk each module, reading chips until the next module's offset.
  // The chip count per module is variable (1, 2 or 4); the 128-bit module-end
  // padding (zero words) ends the walk early via the magic check.
  for (int modIdx = 0; modIdx < numModules; modIdx++) {
    uint32_t detId = detIds[modIdx];
    int moduleStartWord = dataBlockStart + moduleOffsets[modIdx];

    if (moduleStartWord < 0 || moduleStartWord >= fedSizeInWords) {
      edm::LogWarning("RawToBitStreamProducer")
          << "Module offset out of FED bounds: detId=" << detId << " moduleStartWord=" << moduleStartWord
          << " fedSize=" << fedSizeInWords << ". Skipping module.";
      continue;
    }

    // End of this module's data = start of the next module (or the trailer for
    // the last module). Chips are only read within [moduleStartWord, moduleEndWord).
    int moduleEndWord =
        (modIdx + 1 < numModules) ? (dataBlockStart + static_cast<int>(moduleOffsets[modIdx + 1])) : trailerStart;
    if (moduleEndWord > fedSizeInWords)
      moduleEndWord = fedSizeInWords;

    edmNew::DetSetVector<Phase2ITChipBitStream>::FastFiller filler(output, detId);

    int chipCursor = moduleStartWord;
    for (int chipId = 0; chipCursor < moduleEndWord; chipId++) {
      uint32_t chipHeader = readWord(dataPtr, chipCursor);

      uint32_t magic = (chipHeader >> 28) & 0xF;
      // FIXME ignoring the chip error bits with Dummy for now
      uint32_t endBit = (chipHeader >> 16) & 0x1F;
      uint32_t sizeWords = chipHeader & 0xFFFF;

      // A non-magic word inside the module bound is the 128-bit end padding:
      // the module's chips are exhausted, so stop here (not an error).
      if (magic != CHIP_HEADER_MAGIC)
        break;

      // Reconstruct bitstream length in bits.
      //   endBit == 0  -> last word is full or chip is empty: size = sizeWords * 32
      //   endBit  > 0  -> last word holds endBit real bits:   size = (sizeWords - 1) * 32 + endBit
      unsigned int bitstreamSize =
          (endBit == 0) ? (sizeWords * BITS_PER_WORD) : ((sizeWords - 1) * BITS_PER_WORD + endBit);

      std::vector<bool> bitstream = extractBitStream(dataPtr, chipCursor + 1, bitstreamSize, fedSizeInWords);

      Phase2ITChipBitStream chipStream(chipId, bitstream);
      filler.push_back(chipStream);

      chipCursor += 1 + sizeWords;
    }
  }
}

std::string RawToBitStreamProducer::getBitString(const std::vector<bool>& bits, size_t start, size_t len) const {
  std::string result;
  for (size_t i = 0; i < len && start + i < bits.size(); i++) {
    result += (bits[start + i] ? "1" : "0");
    if ((i + 1) % 8 == 0)
      result += " ";
  }
  return result;
}

uint32_t RawToBitStreamProducer::readWord(const unsigned char* dataPtr, int wordIdx) const {
  int byteIdx = wordIdx * 4;
  return (static_cast<uint32_t>(dataPtr[byteIdx]) << 24) | (static_cast<uint32_t>(dataPtr[byteIdx + 1]) << 16) |
         (static_cast<uint32_t>(dataPtr[byteIdx + 2]) << 8) | static_cast<uint32_t>(dataPtr[byteIdx + 3]);
}

std::string RawToBitStreamProducer::wordToHexString(uint32_t word) const {
  std::stringstream ss;
  ss << "0x" << std::hex << std::setw(4) << std::setfill('0') << word;
  return ss.str();
}

std::vector<bool> RawToBitStreamProducer::extractBitStream(const unsigned char* dataPtr,
                                                           int startWord,
                                                           int bitstreamSize,
                                                           int fedSizeInWords) const {
  std::vector<bool> bitstream;
  if (bitstreamSize <= 0)
    return bitstream;

  int fullWords = bitstreamSize / BITS_PER_WORD;
  int remainingBits = bitstreamSize % BITS_PER_WORD;
  int wordsNeeded = fullWords + (remainingBits > 0 ? 1 : 0);

  // Safe guard to ignore when the bits are malformed
  if (startWord < 0 || startWord + wordsNeeded > fedSizeInWords) {
    edm::LogWarning("RawToBitStreamProducer")
        << "Bitstream read out of FED bounds: startWord=" << startWord << ", needs " << wordsNeeded
        << " words, FED size " << fedSizeInWords << " words. Returning empty bitstream.";
    return bitstream;
  }

  bitstream.reserve(bitstreamSize);
  for (int i = 0; i < fullWords; i++) {
    uint32_t word = readWord(dataPtr, startWord + i);
    for (int j = 0; j < BITS_PER_WORD; j++) {
      bool bit = (word >> (31 - j)) & 1;
      bitstream.push_back(bit);
    }
  }
  if (remainingBits > 0) {
    uint32_t word = readWord(dataPtr, startWord + fullWords);
    for (int j = 0; j < remainingBits; j++) {
      bool bit = (word >> (31 - j)) & 1;
      bitstream.push_back(bit);
    }
  }
  if (debug_)
    std::cout << "UNPACKER: First 32 bits of extracted bitstream: " << getBitString(bitstream, 0, 32) << std::endl;
  return bitstream;
}

// FIXME for now this works because we are assuming 4 lines of 0xFFFFFFFF for both headers and trailers
// Later we have to come up with something more concrete to parse out these 4 lines
bool RawToBitStreamProducer::verifyHeaderTrailerPattern(const unsigned char* dataPtr, int wordIdx) const {
  for (int i = 0; i < HEADER_TRAILER_LINES; i++) {
    uint32_t word = readWord(dataPtr, wordIdx + i);
    if (word != HEADER_TRAILER_PATTERN) {
      return false;
    }
  }
  return true;
}

int RawToBitStreamProducer::findTrailerStart(const unsigned char* dataPtr, int fedSizeInWords) const {
  // Start searching from the end, going backwards
  for (int i = fedSizeInWords - HEADER_TRAILER_LINES; i >= HEADER_TRAILER_LINES; --i) {
    if (verifyHeaderTrailerPattern(dataPtr, i)) {
      return i;
    }
  }
  return -1;  // trailer not found
}

DEFINE_FWK_MODULE(RawToBitStreamProducer);
