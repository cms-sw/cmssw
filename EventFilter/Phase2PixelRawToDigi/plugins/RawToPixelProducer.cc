// EDProducer that takes RawDataBuffer and fully decodes it straight to PixelDigi

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITBitReader.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2ITUnpacker.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/SLinkModuleMap.h"

using namespace Phase2DAQFormatSpecification;

class RawToPixelProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  explicit RawToPixelProducer(const edm::ParameterSet&);
  ~RawToPixelProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<RawDataBuffer> fedRawDataToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapBeginRunToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  // Must match the dropTot setting that produced the bitstream. When true the
  // encoded stream omits the per-hit 4-bit ToT field and every digi gets adc=0.
  const bool dropTot_;
  // Must match the encoder's handleGapPixels mode.
  const bool keepMode_;

  std::unique_ptr<SLinkModuleMap> slinkMap_;
  size_t nModules_ = 0;
};

RawToPixelProducer::RawToPixelProducer(const edm::ParameterSet& iConfig)
    : fedRawDataToken_(consumes<RawDataBuffer>(iConfig.getParameter<edm::InputTag>("fedRawDataCollection"))),
      cablingMapBeginRunToken_(
          esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      cablingMapToken_(esConsumes()),
      dropTot_(iConfig.getParameter<bool>("dropTot")),
      keepMode_(
          Phase2ITUnpacker::parseKeepMode(iConfig.getParameter<std::string>("handleGapPixels"), "RawToPixelProducer")) {
  produces<edm::DetSetVector<PixelDigi>>();
}

void RawToPixelProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("fedRawDataCollection", edm::InputTag("rawDataCollector"));
  desc.add<bool>("dropTot", false);
  desc.add<std::string>("handleGapPixels", "DROP");
  descriptions.add("rawToPixelProducer", desc);
}

void RawToPixelProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  slinkMap_ = std::make_unique<SLinkModuleMap>(iSetup.getData(cablingMapBeginRunToken_));
  nModules_ = 0;
  for (const auto& entry : slinkMap_->fedIdToDetIds())
    nModules_ += entry.second.size();
}

void RawToPixelProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<RawDataBuffer> rawBuf;
  iEvent.getByToken(fedRawDataToken_, rawBuf);
  if (!rawBuf.isValid()) {
    throw cms::Exception("RawToPixelProducer") << "Invalid RawDataBuffer";
  }

  auto const& cablingMap = iSetup.getData(cablingMapToken_);

  // Decoded into here and then handed to the DetSetVector constructor that swaps
  // the whole thing in. DetSetVector::insert would instead deep copy every
  // module's digis a second time, which is what its own header warns about.
  std::vector<edm::DetSet<PixelDigi>> moduleSets;
  moduleSets.reserve(nModules_);

  for (const auto& entry : slinkMap_->fedIdToDetIds()) {
    int fedId = entry.first;
    auto frag = rawBuf->fragmentData(static_cast<uint32_t>(fedId));
    if (!frag.isValid()) {
      throw cms::Exception("RawToPixelProducer") << "Missing RawDataBuffer fragment for fed " << fedId
                                                 << ": cabling map lists this FED but the buffer has no source for it.";
    }
    auto fragSpan = frag.data();

    int fedSizeInWords = 0;
    const unsigned char* dataPtr =
        Phase2ITUnpacker::stripSLinkWrapper(fragSpan.data(), fragSpan.size(), fedId, fedSizeInWords);

    const std::vector<uint32_t>& detIds = slinkMap_->detIdsForFedId(fedId);
    if (!Phase2ITUnpacker::verifyHeaderTrailerPattern(dataPtr, 0)) {
      throw cms::Exception("RawToPixelProducer") << "Invalid header in FEDRawData";
    }
    const int trailerStart = Phase2ITUnpacker::findTrailerStart(dataPtr, fedSizeInWords);
    if (trailerStart < 0) {
      throw cms::Exception("RawToPixelProducer") << "Invalid trailer in FEDRawData";
    }

    Phase2ITUnpacker::forEachModule(
        dataPtr, fedSizeInWords, trailerStart, detIds.size(), [&](int modIdx, Phase2ITUnpacker::ModuleSpan span) {
          const uint32_t detId = detIds[modIdx];
          // decoded in place, so the digis are never copied out of here
          auto& moduleDigis = moduleSets.emplace_back(detId);
          // run only for modules that actually carry chips
          int subtype = -1;
          Phase2ITUnpacker::forEachChip(
              dataPtr, span, fedSizeInWords, [&](int chipId, int payloadStartWord, uint32_t nBits) {
                if (subtype < 0)
                  subtype = static_cast<int>(cablingMap.getModuleInfo(detId).subtype);
                Phase2ITBitReader reader(dataPtr + payloadStartWord * BYTES_PER_WORD, nBits);
                Phase2ITUnpacker::decodeChip(reader, chipId, subtype, dropTot_, keepMode_, moduleDigis);
              });
          if (moduleDigis.empty())
            moduleSets.pop_back();
        });
  }

  iEvent.put(std::make_unique<edm::DetSetVector<PixelDigi>>(moduleSets));
}

DEFINE_FWK_MODULE(RawToPixelProducer);
