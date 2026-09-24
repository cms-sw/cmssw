// EDProducer that takes RawDataBuffer and produces ITChipBitStream
// Very first step of unpacker

#include <bitset>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

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
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChipBitStream.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2ITUnpacker.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/SLinkModuleMap.h"

using namespace Phase2DAQFormatSpecification;

class RawToBitStreamProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  explicit RawToBitStreamProducer(const edm::ParameterSet&);
  ~RawToBitStreamProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

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

  for (const auto& entry : slinkMap_->fedIdToDetIds()) {
    int fedId = entry.first;
    auto frag = rawBuf->fragmentData(static_cast<uint32_t>(fedId));
    if (!frag.isValid()) {
      throw cms::Exception("RawToBitStreamProducer")
          << "Missing RawDataBuffer fragment for fed " << fedId
          << ": cabling map lists this FED but the buffer has no source for it.";
    }
    auto fragSpan = frag.data();

    // Hand the IT-specific body (IT header + offsets + data + IT trailer)
    int fedSizeInWords = 0;
    const unsigned char* dataPtr =
        Phase2ITUnpacker::stripSLinkWrapper(fragSpan.data(), fragSpan.size(), fedId, fedSizeInWords);
    processFED(dataPtr, fedSizeInWords, fedId, *output);
  }
  iEvent.put(std::move(output));
}

void RawToBitStreamProducer::processFED(const unsigned char* dataPtr,
                                        int fedSizeInWords,
                                        int fedId,
                                        edmNew::DetSetVector<Phase2ITChipBitStream>& output) {
  const std::vector<uint32_t>& detIds = slinkMap_->detIdsForFedId(fedId);
  if (!Phase2ITUnpacker::verifyHeaderTrailerPattern(dataPtr, 0)) {
    throw cms::Exception("RawToBitStreamProducer") << "Invalid header in FEDRawData";
  }
  int trailerStart = Phase2ITUnpacker::findTrailerStart(dataPtr, fedSizeInWords);
  if (trailerStart < 0) {
    throw cms::Exception("RawToBitStreamProducer") << "Invalid trailer in FEDRawData";
  }

  Phase2ITUnpacker::forEachModule(
      dataPtr, fedSizeInWords, trailerStart, detIds.size(), [&](int modIdx, Phase2ITUnpacker::ModuleSpan span) {
        edmNew::DetSetVector<Phase2ITChipBitStream>::FastFiller filler(output, detIds[modIdx]);
        Phase2ITUnpacker::forEachChip(
            dataPtr, span, fedSizeInWords, [&](int chipId, int payloadStartWord, uint32_t nBits) {
              std::vector<uint8_t> bitstream((nBits + 7) / 8);
              std::memcpy(bitstream.data(), dataPtr + payloadStartWord * BYTES_PER_WORD, bitstream.size());
              if (debug_ && nBits > 0) {
                const std::string bits =
                    std::bitset<32>(Phase2ITUnpacker::readWord(dataPtr, payloadStartWord)).to_string();
                std::cout << "UNPACKER: First 32 bits of extracted bitstream: "
                          << bits.substr(0, std::min<uint32_t>(32u, nBits)) << std::endl;
              }
              filler.push_back(Phase2ITChipBitStream(chipId, std::move(bitstream), nBits));
            });
      });
}

DEFINE_FWK_MODULE(RawToBitStreamProducer);
