// Dump of the BTL RawDataBuffer: SLinkRocket header/trailer and decoded fields of each channel word
#include <bitset>
#include <cstring>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"

#include "EventFilter/MTDRawToDigi/interface/BTLRawDataFormat.h"

class DumpBTLRawDataBuffer : public edm::global::EDAnalyzer<> {
public:
  explicit DumpBTLRawDataBuffer(edm::ParameterSet const&);
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<RawDataBuffer> rawDataBufferToken_;
};

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
DumpBTLRawDataBuffer::DumpBTLRawDataBuffer(edm::ParameterSet const& iPSet)
    : rawDataBufferToken_(consumes<RawDataBuffer>(iPSet.getParameter<edm::InputTag>("rawDataBufferTag"))) {}

// ------------------------------------------------------------
// analyze
// ------------------------------------------------------------
void DumpBTLRawDataBuffer::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
  const auto& rawDataBuffer = iEvent.get(rawDataBufferToken_);

  constexpr std::size_t kHeaderSize = sizeof(SLinkRocketHeader_v3);
  constexpr std::size_t kTrailerSize = sizeof(SLinkRocketTrailer_v3);
  constexpr std::size_t kWordBytes = sizeof(btlraw::ChannelWord);  // 128-bit words

  edm::LogInfo("DumpBTLRawDataBuffer") << "=== Event " << iEvent.id().event()
                                       << " -- number of FED fragments: " << rawDataBuffer.map().size();

  // -- Iterate over all source IDs actually present in the buffer
  for (const auto& entry : rawDataBuffer.map()) {
    const uint32_t sourceId = entry.first;
    const auto& fragData = rawDataBuffer.fragmentData(sourceId);

    if (!fragData.isValid() || fragData.size() == 0) {
      edm::LogWarning("DumpBTLRawDataBuffer") << "No data for source ID = " << sourceId;
      continue;
    }

    edm::LogInfo("DumpBTLRawDataBuffer") << "--- SOURCE ID = " << sourceId << "  fragment size = " << fragData.size()
                                         << " bytes";

    // --------------------------------------------------------
    // SLinkRocket header
    // --------------------------------------------------------
    if (fragData.size() < kHeaderSize) {
      edm::LogWarning("DumpBTLRawDataBuffer")
          << "Fragment too small for S-Link header: " << fragData.size() << " bytes";
      continue;
    }

    const auto headerSpan = fragData.dataHeader(kHeaderSize);
    auto headerView = makeSLinkRocketHeaderView(headerSpan);

    edm::LogInfo("DumpBTLRawDataBuffer") << "  [Header]"
                                         << "  version=" << static_cast<int>(headerView->version())
                                         << "  sourceID=" << headerView->sourceID()
                                         << "  eventID=" << headerView->globalEventID()
                                         << "  l1aTypes=" << headerView->l1aTypes()
                                         << "  l1aPhys=" << static_cast<int>(headerView->l1aPhysType())
                                         << "  emuStatus=" << static_cast<int>(headerView->emuStatus())
                                         << "  markerOK=" << headerView->verifyMarker();

    // --------------------------------------------------------
    // Payload (channel words, 128 bit each)
    // --------------------------------------------------------
    if (fragData.size() < kHeaderSize + kTrailerSize) {
      edm::LogWarning("DumpBTLRawDataBuffer")
          << "Fragment too small for header + trailer: " << fragData.size() << " bytes";
      continue;
    }

    const auto payloadSpan = fragData.payload(kHeaderSize, kTrailerSize);  // return payload between header and trailer
    const std::size_t nWords = payloadSpan.size() / kWordBytes;

    edm::LogInfo("DumpBTLRawDataBuffer") << "  [Payload]  " << nWords << " channel word(s) of 128 bit each";

    // bits from 127 (MSB of hi) down to 0 (LSB of lo)
    auto printWord = [](const btlraw::ChannelWord& w) {
      return std::bitset<64>(w.hi).to_string() + std::bitset<64>(w.lo).to_string();
    };

    for (std::size_t iWord = 0; iWord < nWords; ++iWord) {
      const unsigned char* wordPtr = payloadSpan.data() + iWord * kWordBytes;

      // Copy the 16 bytes of this channel word into a btlraw::ChannelWord
      // and extract each field by name, with the layout defined in BTLRawDataFormat.h
      using namespace btlraw;
      ChannelWord word;
      std::memcpy(&word, wordPtr, kWordBytes);

      const uint64_t BC0count = extractBits(word, kBC0count);
      const uint64_t status = extractBits(word, kStatus);
      const uint64_t slink = extractBits(word, kSLink);
      const uint64_t hslink = extractBits(word, kHSLink);
      const uint64_t elink = extractBits(word, kELink);
      const uint64_t chID = extractBits(word, kChId);
      const uint64_t BCcount = extractBits(word, kBCcount);
      const uint64_t T1coarse = extractBits(word, kT1Coarse);
      const uint64_t T2coarse = extractBits(word, kT2Coarse);
      const uint64_t EOIcoarse = extractBits(word, kEOICoarse);
      const uint64_t Charge = extractBits(word, kCharge);
      const uint64_t T1fine = extractBits(word, kT1Fine);
      const uint64_t T2fine = extractBits(word, kT2Fine);
      const uint64_t IdleTime = extractBits(word, kIdleTime);
      const uint64_t PrevTrigF = extractBits(word, kPrevTrigF);
      const uint64_t TACID = extractBits(word, kTACId);

      edm::LogInfo("DumpBTLRawDataBuffer")
          << "  word " << iWord << "\n"
          << "  bit content: " << printWord(word) << "\n"
          << "  FEDId=" << sourceId << "  BC0count=" << BC0count << "  status=" << status << "  slink=" << slink
          << "  hslink=" << hslink << "  elink=" << elink << "  chID=" << chID << "  BCcount=" << BCcount
          << "  T1coarse=" << T1coarse << "  T2coarse=" << T2coarse << "  EOIcoarse=" << EOIcoarse
          << "  Charge=" << Charge << "  T1fine=" << T1fine << "  T2fine=" << T2fine << "  IdleTime=" << IdleTime
          << "  PrevTrigF=" << PrevTrigF << "  TACID=" << TACID;
    }

    // --------------------------------------------------------
    // SLinkRocket trailer
    // --------------------------------------------------------
    const auto trailerSpan = fragData.dataTrailer(kTrailerSize);
    auto trailerView = makeSLinkRocketTrailerView(trailerSpan, headerView->version());

    edm::LogInfo("DumpBTLRawDataBuffer") << "  [Trailer]"
                                         << "  orbitID=" << trailerView->orbitID() << "  bxID=" << trailerView->bxID()
                                         << "  eventLenBytes=" << trailerView->eventLenBytes()
                                         << "  status=" << trailerView->status() << "  crc=0x" << std::hex
                                         << trailerView->crc() << std::dec << "  daqCRC=0x" << std::hex
                                         << trailerView->daqCRC() << std::dec
                                         << "  markerOK=" << trailerView->verifyMarker();
  }
}

// ------------------------------------------------------------
// fillDescriptions
// ------------------------------------------------------------
void DumpBTLRawDataBuffer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rawDataBufferTag");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(DumpBTLRawDataBuffer);
