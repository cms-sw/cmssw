// -*- C++ -*-
// Package:    EventFilter/Phase2PixelRawToDigi
// Class:      BitStreamToAuroraProducer
// Description: Pack NE events of per-chip RD53 bit-streams into Aurora format
//              and emit one Phase2ITAuroraBitStream per elink of the module.
//              Per-chip Aurora chain: applyBlocking -> orphanPad -> EoS ->
//              header_blocks -> service_blocks.
//
// Author: Si Hyun Jeon, shjeon@cern.ch
//

#include <array>
#include <map>
#include <vector>
#include <cstdint>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChipBitStream.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITAuroraBitStream.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2AuroraPacker.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/ELinkChipMap.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Transition.h"

class BitStreamToAuroraProducer : public edm::one::EDProducer<> {
public:
  explicit BitStreamToAuroraProducer(const edm::ParameterSet&);
  ~BitStreamToAuroraProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<edm::DetSetVector<Phase2ITChipBitStream>> ITChipBitStreamToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  const unsigned int eventsPerStream_;       // NE: events per stream group
  const unsigned int serviceBlockInterval_;  // ND: data blocks per Aurora service block
  unsigned int eventCount_;

  // detId -> chip -> per-event RD53 chip bit streams, accumulated across NE events
  std::map<uint32_t, std::vector<std::vector<std::vector<bool>>>> buffer_;
};

BitStreamToAuroraProducer::BitStreamToAuroraProducer(const edm::ParameterSet& iConfig)
    : ITChipBitStreamToken_(consumes<edm::DetSetVector<Phase2ITChipBitStream>>(
          iConfig.getParameter<edm::InputTag>("Phase2ITChipBitStream"))),
      cablingMapToken_(esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd>()),
      eventsPerStream_(iConfig.getParameter<unsigned int>("eventsPerStream")),
      serviceBlockInterval_(iConfig.getParameter<unsigned int>("serviceBlockInterval")),
      eventCount_(0) {
  using namespace Phase2DAQFormatSpecification;
  if (eventsPerStream_ < (unsigned)AURORA_EVENTS_PER_STREAM_MIN ||
      eventsPerStream_ > (unsigned)AURORA_EVENTS_PER_STREAM_MAX)
    throw cms::Exception("BitStreamToAuroraProducer")
        << "eventsPerStream=" << eventsPerStream_ << " out of range [" << AURORA_EVENTS_PER_STREAM_MIN << ", "
        << AURORA_EVENTS_PER_STREAM_MAX << "]";
  if (serviceBlockInterval_ < (unsigned)AURORA_SERVICE_BLOCK_INTERVAL_MIN ||
      serviceBlockInterval_ > (unsigned)AURORA_SERVICE_BLOCK_INTERVAL_MAX)
    throw cms::Exception("BitStreamToAuroraProducer")
        << "serviceBlockInterval=" << serviceBlockInterval_ << " out of range [" << AURORA_SERVICE_BLOCK_INTERVAL_MIN
        << ", " << AURORA_SERVICE_BLOCK_INTERVAL_MAX << "]";
  produces<edm::DetSetVector<Phase2ITAuroraBitStream>>();
  produces<bool>("isComplete");
}

void BitStreamToAuroraProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Phase2ITChipBitStream", edm::InputTag("Phase2ITQCoreProducer"));
  using Phase2DAQFormatSpecification::AURORA_EVENTS_PER_STREAM_DEFAULT;
  using Phase2DAQFormatSpecification::AURORA_SERVICE_BLOCK_INTERVAL_DEFAULT;
  desc.add<unsigned int>("eventsPerStream", AURORA_EVENTS_PER_STREAM_DEFAULT);            // NE, 1..64
  desc.add<unsigned int>("serviceBlockInterval", AURORA_SERVICE_BLOCK_INTERVAL_DEFAULT);  // ND, 1..256
  descriptions.add("BitStreamToAuroraProducer", desc);
}

void BitStreamToAuroraProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<DetSetVector<Phase2ITChipBitStream>> handle;
  iEvent.getByToken(ITChipBitStreamToken_, handle);
  if (!handle.isValid())
    throw cms::Exception("BitStreamToAuroraProducer") << "Invalid BitStream handle";

  // Accumulate this event's chip bit streams into the buffer.
  for (const auto& detset : *handle) {
    auto& moduleBuffer = buffer_[detset.id];
    if (moduleBuffer.empty())
      moduleBuffer.resize(detset.size());
    unsigned int chipIdx = 0;
    for (const auto& chipBitStream : detset) {
      moduleBuffer[chipIdx].push_back(chipBitStream.get_bitstream());
      chipIdx++;
    }
  }

  eventCount_++;
  const bool complete = (eventCount_ % eventsPerStream_ == 0);

  if (complete) {
    const auto& cablingMap = iSetup.getData(cablingMapToken_);

    constexpr int EVENT_ID_PLACEHOLDER = 1;  // TODO: drive from iEvent.id().event() / NE-group base

    std::vector<edm::DetSet<Phase2ITAuroraBitStream>> detsets;
    for (auto& [detId, chips] : buffer_) {
      const auto& info = cablingMap.getModuleInfo(detId);
      const unsigned int nChipsSeen = chips.size();
      if (info.nChips != nChipsSeen) {
        edm::LogWarning("BitStreamToAuroraProducer")
            << "detId=" << detId << ": cabling nChips=" << static_cast<int>(info.nChips) << " disagrees with "
            << nChipsSeen << " chip streams seen";
      }

      const ELinkChipMap::ChipToElinks fanout = ELinkChipMap::resolveFanout(static_cast<int>(info.subtype));
      const int nElinks = ELinkChipMap::numElinks(fanout);

      // Per-chip Aurora chain. The chip index is used as the RD53 chip bus ID
      // (2-bit field in every Aurora block, multi-chip mode for all modules).
      std::vector<std::vector<bool>> chipFullBits(nChipsSeen);
      for (unsigned int c = 0; c < nChipsSeen; ++c) {
        std::vector<bool> concat;
        for (unsigned int e = 0; e < chips[c].size(); ++e) {
          auto tag = Phase2AuroraPacker::makeEventTag(EVENT_ID_PLACEHOLDER, (e == 0));
          concat.insert(concat.end(), tag.begin(), tag.end());
          concat.insert(concat.end(), chips[c][e].begin(), chips[c][e].end());
        }
        // TODO for now we are applying the chip id blockings by default although it might be not needed for some elinks!!!
        auto blocked = Phase2AuroraPacker::applyBlocking(concat, static_cast<int>(c));
        auto padded = Phase2AuroraPacker::orphanPad(blocked);
        auto eos = Phase2AuroraPacker::applyEosMarker(padded);
        auto headered = Phase2AuroraPacker::applyHeaderBlocks(eos);
        auto full = Phase2AuroraPacker::applyServiceBlocks(headered, serviceBlockInterval_);
        chipFullBits[c] = std::move(full);
      }

      std::vector<std::vector<bool>> elinkBits(nElinks);
      for (unsigned int c = 0; c < nChipsSeen; ++c) {
        const std::vector<int>& chipElinks = fanout[c];
        const size_t k = chipElinks.size();
        const std::vector<bool>& bits = chipFullBits[c];
        const size_t n = bits.size();
        size_t pos = 0;
        for (size_t i = 0; i < k; ++i) {
          // Even split: the first (n % k) lanes get one extra bit.
          const size_t cnt = n / k + (i < n % k ? 1 : 0);
          auto& lane = elinkBits[chipElinks[i]];
          lane.insert(lane.end(), bits.begin() + pos, bits.begin() + pos + cnt);
          pos += cnt;
        }
      }

      // One Phase2ITAuroraBitStream per elink.
      edm::DetSet<Phase2ITAuroraBitStream> detset(detId);
      for (int e = 0; e < nElinks; ++e) {
        Phase2ITAuroraBitStream aurora(e, eventsPerStream_);
        aurora.addAuroraStream(elinkBits[e]);
        detset.push_back(std::move(aurora));
      }
      detsets.push_back(std::move(detset));
    }

    auto output = std::make_unique<edm::DetSetVector<Phase2ITAuroraBitStream>>(detsets);
    iEvent.put(std::move(output));
    buffer_.clear();
  }

  iEvent.put(std::make_unique<bool>(complete), "isComplete");
}

DEFINE_FWK_MODULE(BitStreamToAuroraProducer);
