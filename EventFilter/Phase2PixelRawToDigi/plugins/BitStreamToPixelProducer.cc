// EDProducer that takes ITChipBitStream and fully decodes it back to PixelDigi
// Second and final step of unpacker

#include <memory>
#include <vector>
#include <algorithm>
#include <array>
#include <functional>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"  // input bitstream uses new container
#include "DataFormats/Common/interface/DetSetVector.h"     // output bitstream uses old container

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChipBitStream.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2ITUnpacker.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

using namespace Phase2DAQFormatSpecification;

class BitStreamToPixelProducer : public edm::stream::EDProducer<> {
public:
  explicit BitStreamToPixelProducer(const edm::ParameterSet&);
  ~BitStreamToPixelProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<edmNew::DetSetVector<Phase2ITChipBitStream>> bitstreamToken_;
  // Cabling map supplies the per-module Module_SubType that keys the ChipModuleMap
  // chip-index convention (must match the packer).
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  // Must match the dropTot setting that produced the bitstream. When true the encoded stream omits the per-hit 4-bit ToT field.
  // the decoder skips decodeADCs and emits PixelDigi.adc = 0 for every hit.
  const bool dropTot_;
  // Must match the encoder's handleGapPixels mode.
  // KEEP shifts each chip boundary to the gap midline so former gap pixels reverse correctly.
  // DROP/AGGREGATE use the standard physical chip extents.
  const bool keepMode_;
};

BitStreamToPixelProducer::BitStreamToPixelProducer(const edm::ParameterSet& iConfig)
    : bitstreamToken_(consumes<edmNew::DetSetVector<Phase2ITChipBitStream>>(
          iConfig.getParameter<edm::InputTag>("phase2ItChipBitStream"))),
      cablingMapToken_(esConsumes()),
      dropTot_(iConfig.getParameter<bool>("dropTot")),
      keepMode_(Phase2ITUnpacker::parseKeepMode(iConfig.getParameter<std::string>("handleGapPixels"),
                                                "BitStreamToPixelProducer")) {
  produces<edm::DetSetVector<PixelDigi>>();
}

void BitStreamToPixelProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("phase2ItChipBitStream", edm::InputTag("rawToBitStreamProducer"));
  desc.add<bool>("dropTot", false);
  desc.add<std::string>("handleGapPixels", "DROP");
  descriptions.add("bitstreamToPixelProducer", desc);
}

void BitStreamToPixelProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto outputPixelDigis = std::make_unique<edm::DetSetVector<PixelDigi>>();
  edm::Handle<edmNew::DetSetVector<Phase2ITChipBitStream>> bitstreamHandle;
  iEvent.getByToken(bitstreamToken_, bitstreamHandle);
  if (!bitstreamHandle.isValid()) {
    throw cms::Exception("BitStreamToPixelProducer") << "Invalid BitStream handle";
  }

  auto const& cablingMap = iSetup.getData(cablingMapToken_);

  // Loop over each DetSet in the input bitstream collection
  // Accumulate every chip's hits into a single per-module detSet, then insert once
  for (const auto& detSet : *bitstreamHandle) {
    DetId tkId = detSet.id();
    uint32_t detId = tkId.rawId();
    // Module_SubType keys the ChipModuleMap convention used to invert chipId -> (row, col) offset; must match what the packer used.
    const int subtype = static_cast<int>(cablingMap.getModuleInfo(detId).subtype);
    edm::DetSet<PixelDigi> moduleDigis(detId);
    for (const auto& chipBS : detSet) {
      Phase2ITBitReader reader(chipBS.bytes().data(), chipBS.nBits());
      Phase2ITUnpacker::decodeChip(reader, chipBS.get_rocid(), subtype, dropTot_, keepMode_, moduleDigis);
    }
    if (!moduleDigis.empty())
      outputPixelDigis->insert(moduleDigis);
  }

  iEvent.put(std::move(outputPixelDigis));
}

DEFINE_FWK_MODULE(BitStreamToPixelProducer);
