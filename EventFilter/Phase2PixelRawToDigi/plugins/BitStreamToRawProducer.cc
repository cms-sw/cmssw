// EDProducer that takes bitstream to RawDataBuffer (Packer)

#include <memory>
#include <new>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"

#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChipBitStream.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/SLinkModuleMap.h"

using namespace Phase2DAQFormatSpecification;

class BitStreamToRawProducer : public edm::one::EDProducer<edm::one::WatchRuns> {
public:
  explicit BitStreamToRawProducer(const edm::ParameterSet&);
  ~BitStreamToRawProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  const edm::EDGetTokenT<edm::DetSetVector<Phase2ITChipBitStream>> ITChipBitStreamToken_;

  void addWordToBuffer(unsigned char* buffer, size_t position, uint32_t word);
  void addWordToBitVector(std::vector<bool>& vec, uint32_t word);
  void padToChunkBoundary(std::vector<bool>& vec);

  std::unique_ptr<SLinkModuleMap> slinkMap_;
};

BitStreamToRawProducer::BitStreamToRawProducer(const edm::ParameterSet& iConfig)
    : cablingMapToken_(
          esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      ITChipBitStreamToken_(consumes<edm::DetSetVector<Phase2ITChipBitStream>>(
          iConfig.getParameter<edm::InputTag>("Phase2ITChipBitStream"))) {
  produces<RawDataBuffer>();
}

void BitStreamToRawProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Phase2ITChipBitStream", edm::InputTag("PixelQCoreProducer"));
  descriptions.add("pixelToRawProducer", desc);
}

void BitStreamToRawProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  slinkMap_ = std::make_unique<SLinkModuleMap>(iSetup.getData(cablingMapToken_));
}

void BitStreamToRawProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  edm::Handle<edm::DetSetVector<Phase2ITChipBitStream>> handle;
  iEvent.getByToken(ITChipBitStreamToken_, handle);

  if (!handle.isValid()) {
    throw cms::Exception("BitStreamToRawProducer") << "Invalid BitStream handle";
  }

  // Per-fragment scratch: cached offset/data bit-vectors + total bytes including
  // SLinkRocket header/trailer + IT header/trailer placeholders.
  struct FedFrame {
    int fedId;
    std::vector<bool> offsetBlock;  // already padded to 128-bit
    std::vector<bool> dataBlock;    // already padded to 128-bit
    unsigned int totalSize;         // bytes
  };
  std::vector<FedFrame> frames;
  frames.reserve(slinkMap_->fedIdToDetIds().size());

  // Each fragment carries:
  //   SLinkRocket header (16 B)
  // + IT header placeholder (16 B = HEADER_TRAILER_LINES words)
  // + offset table (128-bit padded)
  // + data block (128-bit padded)
  // + IT trailer placeholder (16 B)
  // + SLinkRocket trailer (16 B)
  const unsigned int SLINK_HDR_BYTES = sizeof(SLinkRocketHeader_v3);
  const unsigned int SLINK_TRL_BYTES = sizeof(SLinkRocketTrailer_v3);
  const unsigned int IT_HDR_BYTES    = HEADER_TRAILER_LINES * BYTES_PER_WORD;
  const unsigned int IT_TRL_BYTES    = HEADER_TRAILER_LINES * BYTES_PER_WORD;

  // Build bit blocks per FED and compute totalSize=totalRawDataBuffer.
  uint32_t totalRawDataBuffer = 0;
  for (const auto& entry : slinkMap_->fedIdToDetIds()) {
    FedFrame f;
    f.fedId = entry.first;
    const std::vector<uint32_t>& detIds = entry.second;

    for (uint32_t detId : detIds) {
      auto foundDetId = handle->find(detId);
      if (foundDetId == handle->end()) {
        throw cms::Exception("BitStreamToRawProducer") << "Could not find detId from the inputs";
      }
      const edm::DetSet<Phase2ITChipBitStream>& detSet = *foundDetId;

      // Block 2: per-module offsets.
      // Module-level offset = dataBlock position for module in 32-bit words.
      // FIXME For the first module the value is always 0, is it necessary?
      uint32_t moduleOffset = f.dataBlock.size() / BITS_PER_WORD;
      addWordToBitVector(f.offsetBlock, moduleOffset);

      // Block 3: per-module sequence of (chip header word, chip bitstream, 32-bit pad)
      // repeated for CHIPS_PER_MODULE chips.
      for (auto const& chip : detSet) {
        std::vector<bool> chipBitStream = chip.get_bitstream();
        unsigned int bitstreamSize = chipBitStream.size();
        unsigned int sizeWords = (bitstreamSize + BITS_PER_WORD - 1) / BITS_PER_WORD;
        unsigned int endBit = bitstreamSize % BITS_PER_WORD;

        // Per-chip header word:
        //   bits 31..28 : magic = 0xE
        //   bits 27..24 : error flags FIXME dummy given for now
        //   bits 23..21 : reserved
        //   bits 20..16 : end bit number (bits used in last 32-bit word)
        //   bits 15..0  : chip bitstream size in 32-bit words, NOT PER MODULE
        uint32_t chipHeader = ((CHIP_HEADER_MAGIC & 0xF) << 28) | ((0u & 0xF) << 24) | ((0u & 0x7) << 21) |
                              ((endBit & 0x1F) << 16) | (sizeWords & 0xFFFF);
        addWordToBitVector(f.dataBlock, chipHeader);
        f.dataBlock.insert(f.dataBlock.end(), chipBitStream.begin(), chipBitStream.end());

        // Pad chip bitstream to 32-bit boundary
        unsigned int chipPadBits = (endBit > 0) ? (BITS_PER_WORD - endBit) : 0;
        if (chipPadBits > 0) {
          f.dataBlock.insert(f.dataBlock.end(), chipPadBits, false);
        }
      }

      // Pad module to 128-bit boundary at module end
      padToChunkBoundary(f.dataBlock);
    }

    // Pad offset block to 128-bit boundary
    padToChunkBoundary(f.offsetBlock);

    unsigned int offsetSize = f.offsetBlock.size() / BITS_PER_WORD * BYTES_PER_WORD;
    unsigned int dataSize   = f.dataBlock.size()   / BITS_PER_WORD * BYTES_PER_WORD;
    f.totalSize = SLINK_HDR_BYTES + IT_HDR_BYTES + offsetSize + dataSize + IT_TRL_BYTES + SLINK_TRL_BYTES;
    totalRawDataBuffer += f.totalSize;
    frames.push_back(std::move(f));
  }

  // Allocate one RawDataBuffer big enough to hold every fragment,
  // then fill each one in-place via addSource(srcId, nullptr, totalSize).
  auto raw = std::make_unique<RawDataBuffer>(totalRawDataBuffer);

  const auto& aux = iEvent.eventAuxiliary();
  const uint64_t eventId = aux.event();
  const uint32_t orbitId = aux.orbitNumber();
  const uint16_t bxId    = static_cast<uint16_t>(aux.bunchCrossing());

  for (auto& f : frames) {
    unsigned char* buffer = raw->addSource(f.fedId, nullptr, f.totalSize);

    // SLinkRocket header (16 B) at the very start.
    const uint16_t l1a_types  = 1;
    const uint8_t  l1a_phys   = 0;
    const uint8_t  emu_status = 2;  // DTH emulator
    new ((void*)buffer)
        SLinkRocketHeader_v3(static_cast<uint32_t>(f.fedId), l1a_types, l1a_phys, emu_status, eventId);

    // Word index within the fragment (4-byte words) after the SLinkRocket header.
    unsigned int wordIdx = SLINK_HDR_BYTES / BYTES_PER_WORD;  // = 4

    // IT header placeholder (4 lines of 0xFFFFFFFF) FIXME Dummy given for now
    for (int i = 0; i < HEADER_TRAILER_LINES; i++) {
      addWordToBuffer(buffer, wordIdx++, HEADER_TRAILER_PATTERN);
    }

    // Offset block
    unsigned int offsetWords = f.offsetBlock.size() / BITS_PER_WORD;
    for (unsigned int i = 0; i < offsetWords; i++) {
      uint32_t word = 0;
      for (int bit = 0; bit < BITS_PER_WORD; bit++) {
        if (f.offsetBlock[i * BITS_PER_WORD + bit]) {
          word |= (1 << (31 - bit));
        }
      }
      addWordToBuffer(buffer, wordIdx++, word);
    }

    // Data block
    unsigned int dataWords = f.dataBlock.size() / BITS_PER_WORD;
    for (unsigned int i = 0; i < dataWords; i++) {
      uint32_t word = 0;
      for (int bit = 0; bit < BITS_PER_WORD; bit++) {
        if (f.dataBlock[i * BITS_PER_WORD + bit]) {
          word |= (1 << (31 - bit));
        }
      }
      addWordToBuffer(buffer, wordIdx++, word);
    }

    // IT trailer placeholder (4 lines of 0xFFFFFFFF) FIXME dummy given for now
    for (int i = 0; i < HEADER_TRAILER_LINES; i++) {
      addWordToBuffer(buffer, wordIdx++, HEADER_TRAILER_PATTERN);
    }

    // SLinkRocket trailer (16 B) at the very end. event_length_wcount is the
    // fragment size in 16-byte SLinkRocket words.
    const uint16_t status  = 0;
    const uint16_t crc     = 0;
    const uint16_t daq_crc = 0;
    const uint32_t evtLenWc = f.totalSize >> SLR_WORD_NUM_BYTES_SHIFT;
    new ((void*)(buffer + f.totalSize - SLINK_TRL_BYTES))
        SLinkRocketTrailer_v3(status, crc, orbitId, bxId, evtLenWc, daq_crc);
  }

  iEvent.put(std::move(raw));
}

void BitStreamToRawProducer::addWordToBuffer(unsigned char* buffer, size_t position, uint32_t word) {
  buffer[position * 4]     = (word >> 24) & 0xFF;
  buffer[position * 4 + 1] = (word >> 16) & 0xFF;
  buffer[position * 4 + 2] = (word >>  8) & 0xFF;
  buffer[position * 4 + 3] =  word        & 0xFF;
}

// Append a 32-bit word to a bit vector
void BitStreamToRawProducer::addWordToBitVector(std::vector<bool>& vec, uint32_t word) {
  for (int bit = 31; bit >= 0; bit--) {
    vec.push_back((word >> bit) & 1);
  }
}

// Pad bit vector with zeros up to the next 128-bit chunk boundary
void BitStreamToRawProducer::padToChunkBoundary(std::vector<bool>& vec) {
  if (!vec.empty() && vec.size() % BITS_PER_CHUNK != 0) {
    size_t paddingNeeded = BITS_PER_CHUNK - (vec.size() % BITS_PER_CHUNK);
    vec.insert(vec.end(), paddingNeeded, false);
  }
}

DEFINE_FWK_MODULE(BitStreamToRawProducer);
